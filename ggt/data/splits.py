#!/usr/bin/env python
"""Seeded, persistent train/devel/test splits and the fitted label scaler.

The contract, and why this replaced the original `make_splits`:

* **A split is created once and then reused.** The original hardcoded
  `random_state=0`, offered no seed option, and rewrote 14 slug variants on
  every invocation. Here an existing split is loaded and its manifest hash
  verified; changing it requires `--force-resplit` AND a new seed, so a
  hyperparameter sweep can never silently regenerate the test set.
* **The split is checked against the current `info.csv`.** Hashes alone
  only prove a CSV is unedited. Rebuilding `info.csv` leaves the old split
  self-consistent and completely wrong, so the union of its object_ids must
  still equal the catalog's.
* **The scaler is fitted once, on the train split only, and persisted.**
  `standardize_labels` refits from the train CSV on every dataset
  construction. That is deterministic given a fixed split, so it is not
  *wrong* -- but it is invisible, unverifiable, and changes the moment
  anything touches the split.

    python -m ggt.data.splits --z-bin 0 --band VIS --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from ggt.data import layout
from ggt.surveys import euclid

log = logging.getLogger("ggt.data.splits")

DEFAULT_FRACTIONS = {"train": 0.70, "devel": 0.15, "test": 0.15}


def slug_for(seed: int) -> str:
    return f"euclid-{seed}"


def _hash_ids(ids) -> str:
    arr = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _interleave(groups):
    """Round-robin the group index lists."""
    from itertools import chain, zip_longest

    return [x for x in chain(*zip_longest(*groups)) if x is not None]


def make_split(
    df: pd.DataFrame, seed: int, fractions=None, balance_on="bt", n_bins=4
):
    """Assign each row to train/devel/test.

    With `balance_on` set, rows are first partitioned into `n_bins`
    quantile bins of that column and then interleaved, so each split spans
    the full range of the balancing variable rather than over-representing
    whatever the shuffle happened to put first. This is the original
    `balanced` semantics, expressed on quantiles so that it behaves on a
    skewed distribution.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise SystemExit(f"fractions must sum to 1, got {total}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))

    if balance_on and balance_on in df.columns and len(df) >= n_bins * 4:
        vals = df[balance_on].to_numpy()[order]
        edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
        groups = [
            list(order[np.digitize(vals, edges) == b]) for b in range(n_bins)
        ]
        order = np.array(_interleave([g for g in groups if g]))

    assign = np.empty(len(df), dtype=object)
    start = 0
    for name, frac in fractions.items():
        stop = start + int(np.ceil(len(df) * frac))
        assign[order[start:stop]] = name
        start = stop
    assign[assign == None] = "test"  # noqa: E711 -- rounding leftovers
    return assign


def build(
    z_bin,
    band,
    seed=42,
    fractions=None,
    balance_on="bt",
    root=None,
    force_resplit=False,
    target_metrics=None,
):
    from joblib import dump
    from sklearn.preprocessing import StandardScaler

    target_metrics = list(target_metrics or euclid.TARGET_COLUMNS)
    info_path = layout.info_csv(z_bin, band, root)
    if not info_path.exists():
        raise SystemExit(f"{info_path} not found; run build_info first")
    df = pd.read_csv(info_path)

    slug = slug_for(seed)
    sdir = layout.splits_dir(z_bin, band, root)
    mpath = layout.split_manifest(z_bin, band, slug, root)
    files = {s: sdir / f"{slug}-{s}.csv" for s in ("train", "devel", "test")}
    scaler_path = layout.scaler_path(z_bin, band, slug, root)

    # --- reuse an existing split unless explicitly told otherwise ---
    if all(f.exists() for f in files.values()) and mpath.exists():
        man = json.loads(mpath.read_text())
        if not force_resplit:
            sizes, ok = {}, True
            for s, f in files.items():
                ids = pd.read_csv(f)["object_id"].to_numpy()
                sizes[s] = len(ids)
                if _hash_ids(ids) != man["hashes"][s]:
                    ok = False
                    log.error(
                        "%s no longer matches its manifest hash -- the "
                        "split on disk has "
                        "been edited. Refusing to proceed.",
                        f,
                    )
            if not ok:
                raise SystemExit(
                    "split/manifest mismatch; investigate before training"
                )

            # The hashes above only prove each CSV is unedited -- they
            # say nothing about whether the split still describes THIS
            # info.csv. Rebuilding info.csv (a retuned cut, or debug ->
            # production data) leaves the old split perfectly
            # self-consistent and completely wrong, and the run would
            # train on the stale subset while logging "reusing existing
            # split".
            info_ids = set(pd.read_csv(info_path)["object_id"].tolist())
            split_ids = set()
            for f in files.values():
                split_ids.update(pd.read_csv(f)["object_id"].tolist())
            if split_ids != info_ids:
                raise SystemExit(
                    f"the {slug} split does not describe the current "
                    f"{info_path.name}:\n"
                    f"  info.csv  : {len(info_ids)} sources\n"
                    f"  {slug} split: {len(split_ids)} sources "
                    f"({len(info_ids - split_ids)} missing from the split, "
                    f"{len(split_ids - info_ids)} not in info.csv)\n"
                    "info.csv has been rebuilt since this split was "
                    "made. Choose a new "
                    "--seed, or move the stale split files aside."
                )

            log.info(
                "reusing existing split %s (train/devel/test = %s, hash %s)",
                slug,
                "/".join(str(sizes[s]) for s in ("train", "devel", "test")),
                man["hashes"]["train"],
            )
            if scaler_path.exists():
                log.info("reusing existing scaler %s", scaler_path)
                return man
            log.warning(
                "scaler missing; refitting from the existing train split"
            )
        else:
            # The manifest is per-slug, so reaching here means the SAME
            # seed is being regenerated -- which would overwrite a split
            # that old runs were trained against.
            raise SystemExit(
                f"--force-resplit with seed {seed} would overwrite the "
                f"existing {slug} split in place, making runs trained on "
                "it uninterpretable. Choose a new --seed "
                "instead; different seeds coexist."
            )

    layout.ensure(sdir)
    assign = make_split(df, seed, fractions, balance_on)
    hashes, sizes = {}, {}
    for s, f in files.items():
        part = df[assign == s]
        part.to_csv(f, index=False)
        hashes[s] = _hash_ids(part["object_id"])
        sizes[s] = len(part)
    log.info(
        "split %s: train/devel/test = %d/%d/%d",
        slug,
        sizes["train"],
        sizes["devel"],
        sizes["test"],
    )

    # --- scaler: fitted on the TRAIN split only, then persisted ---
    train = df[assign == "train"]
    scaler = StandardScaler().fit(
        np.asarray(train[target_metrics], dtype=float)
    )
    dump(scaler, scaler_path)
    for name, mean, scale in zip(target_metrics, scaler.mean_, scaler.scale_):
        log.info("  scaler %-22s mean %8.4f  scale %8.4f", name, mean, scale)

    manifest = {
        "slug": slug,
        "seed": seed,
        "z_bin": z_bin,
        "band": band,
        "fractions": fractions or DEFAULT_FRACTIONS,
        "balance_on": balance_on,
        "n_total": len(df),
        "sizes": sizes,
        "hashes": hashes,
        "target_metrics": target_metrics,
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    }
    mpath.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_scaler(z_bin, band, seed=42, root=None, target_metrics=None):
    """Load the persisted scaler, verifying the expected target order."""
    from joblib import load

    p = layout.scaler_path(z_bin, band, slug_for(seed), root)
    if not p.exists():
        raise SystemExit(
            f"{p} not found; run `python -m ggt.data.splits` first"
        )
    scaler = load(p)
    expected = list(target_metrics or euclid.TARGET_COLUMNS)
    if scaler.mean_.shape[0] != len(expected):
        raise SystemExit(
            f"scaler was fitted on {scaler.mean_.shape[0]} targets but "
            f"{len(expected)} were "
            f"requested ({expected}); refit or fix --target-metrics"
        )
    return scaler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--z-bin", type=int, required=True)
    ap.add_argument("--band", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--devel-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--balance-on", default="bt")
    ap.add_argument("--force-resplit", action="store_true")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    build(
        a.z_bin,
        a.band,
        a.seed,
        {"train": a.train_frac, "devel": a.devel_frac, "test": a.test_frac},
        a.balance_on,
        a.out_root,
        a.force_resplit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
