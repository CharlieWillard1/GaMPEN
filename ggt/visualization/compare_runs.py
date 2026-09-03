#!/usr/bin/env python
"""Compare several finished runs on the same held-out split.

    python -m ggt.visualization.compare_runs OUT_DIR RUN_DIR [RUN_DIR ...]

This is the tier that cannot run at the end of a single training run,
because every question it answers is a question about a *set* of runs: which
dropout rate is calibrated, whether the real-HSC checkpoint beats the
simulated one, whether freezing the VGG features helps, whether the 400 px
native geometry costs anything against resampling onto HSC's grid, whether
`--pixel-zp 27` was worth a second export, and how performance scales with
the number of GALFIT labels.

**It never runs inference itself.** Each run must already have a
`posteriors_<split>/` cache written by `posterior_figures`. That keeps this
module cheap and re-runnable, and it means a comparison can never silently
be made against a stale or differently-parameterised inference pass.

The one comparison that is not safe by construction is amplitude: two runs
trained at different `--pixel-zp` see pixels ~9x apart and are comparable
only in unscaled label space. `check_comparable` says so out loud rather
than letting the difference be read as a result.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ggt.visualization import coverage as cov  # noqa: E402
from ggt.visualization import posterior_figures as pf  # noqa: E402
from ggt.visualization import style  # noqa: E402

log = logging.getLogger(__name__)

_save = style.save

#: Config keys worth grouping runs by, and how to label the resulting axis.
COMPARISON_KEYS = {
    "dropout": "dropout rate",
    "init_from": "initialisation",
    "freeze": "freeze strategy",
    "resolved_cutout_size": "input size [px]",
    "pixel_zp": "pixel zeropoint",
    "lr": "learning rate",
}


def load_run(run_dir, split="devel"):
    """One run's config, cached posteriors and GALFIT truths.

    Returns `None` (with a warning) when the run has no cache, so a
    comparison over a directory of runs skips the ones not yet analysed
    instead of failing outright.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    cached = pf.read_cache(pf.cache_dir(run_dir, split))
    if cached is None:
        log.warning(
            "%s has no posteriors_%s cache -- run posterior_figures on it "
            "first; skipping", run_dir.name, split,
        )
        return None
    df, samples = cached
    truths = {
        c: np.asarray(df[c], dtype=float)
        for c in set(pf._TRUTH_COLUMN.values())
        if c in df.columns
    }
    return {
        "name": cfg.get("run_name") or run_dir.name,
        "run_dir": run_dir,
        "cfg": cfg,
        "df": df,
        "samples": samples,
        "truths": truths,
        "n_train": int((cfg.get("split_sizes") or {}).get("train", 0)),
    }


def check_comparable(runs):
    """Warn about the differences that make a comparison meaningless.

    Pixel zeropoint is the one that bites: runs exported at different
    `--pixel-zp` were trained on pixels roughly 9x apart in amplitude, and
    any difference in their *pixel-space* behaviour is that, not the thing
    being tested. Different splits or z-bins are worse -- those are simply
    different galaxies.
    """
    notes = []
    for key, why in (
        ("pixel_zp", "pixels differ in amplitude; compare only unscaled labels"),
        ("cache_pixel_zp", "the cached stamps themselves differ"),
        ("z_bin", "different galaxies entirely"),
        ("band", "different galaxies entirely"),
        ("seed", "different train/devel/test partitions"),
    ):
        values = {json.dumps(r["cfg"].get(key)) for r in runs}
        if len(values) > 1:
            note = f"{key} differs across runs ({', '.join(sorted(values))}): {why}"
            log.warning("NOT DIRECTLY COMPARABLE: %s", note)
            notes.append(note)
    return notes


# --- metrics -----------------------------------------------------------------


def run_metrics(run, percentiles=cov.DEFAULT_PERCENTILES):
    """Per-parameter accuracy and calibration for one run.

    Residual sigma uses the median absolute deviation scaled to a Gaussian
    rather than the plain standard deviation: a handful of catastrophic
    outliers should not decide which of two runs looks better.
    """
    df, truths, samples = run["df"], run["truths"], run["samples"]
    rows = []
    for col in pf.PHYSICAL_COLUMNS:
        tcol = pf._TRUTH_COLUMN.get(col)
        if f"{col}_mode" not in df.columns or tcol not in truths:
            continue
        pred = np.asarray(df[f"{col}_mode"], dtype=float)
        r = pf._residual(col, pred, truths[tcol])
        r = r[np.isfinite(r)]

        entry = {
            "run": run["name"],
            "parameter": col,
            "n": int(r.size),
            "bias_median": float(np.median(r)) if r.size else np.nan,
            "sigma_std": float(np.std(r)) if r.size else np.nan,
            "sigma_nmad": (
                float(1.4826 * np.median(np.abs(r - np.median(r))))
                if r.size else np.nan
            ),
        }
        if col in samples:
            try:
                achieved, _, _ = cov.coverage_curve(
                    samples[col], truths[tcol], percentiles=percentiles
                )
                entry["coverage_68"] = float(
                    achieved[list(percentiles).index(0.6827)]
                )
                entry["coverage_dev"] = cov.deviation_from_diagonal(
                    percentiles, achieved
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("%s/%s: coverage failed (%s)", run["name"], col, exc)
        rows.append(entry)
    return rows


def summary_table(runs, out_dir, split="devel"):
    """The one table the whole comparison reduces to. Written as CSV."""
    rows = []
    for run in runs:
        rows.extend(run_metrics(run))
    df = pd.DataFrame(rows)
    for key in COMPARISON_KEYS:
        df[key] = [
            next(r["cfg"].get(key) for r in runs if r["name"] == name)
            for name in df["run"]
        ]
    df["n_train"] = [
        next(r["n_train"] for r in runs if r["name"] == name) for name in df["run"]
    ]
    path = Path(out_dir) / f"comparison_{split}.csv"
    df.to_csv(path, index=False)
    log.info("wrote %s", path)
    return df


# --- figures -----------------------------------------------------------------


def coverage_sweep(runs, out_dir, split="devel", key="dropout"):
    """G22 Fig. 8 / Table 2 swept over runs -- the dropout-tuning figure.

    Ghosh's guidance is to tune dropout last, after every other
    hyperparameter, and to pick the rate whose mean curve tracks the
    diagonal. That is exactly what this shows: one mean coverage curve per
    run, labelled by the value of `key`, with the mean deviation in the
    legend as the number to sort on.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")

    best = None
    for run in runs:
        by_target, truth_by_target = {}, {}
        for col in pf.PHYSICAL_COLUMNS:
            tcol = pf._TRUTH_COLUMN.get(col)
            if col in run["samples"] and tcol in run["truths"]:
                by_target[col] = run["samples"][col]
                truth_by_target[col] = run["truths"][tcol]
        if not by_target:
            continue
        pcts, _, mean = cov.coverage_table(by_target, truth_by_target)
        dev = cov.deviation_from_diagonal(pcts, mean)
        label = f"{run['cfg'].get(key)}  ({dev:.1f}%)"
        ax.plot(pcts, mean, marker="o", ms=3.5, lw=1.4, label=label)
        if best is None or dev < best[1]:
            best = (run["name"], dev, run["cfg"].get(key))

    ax.set_xlabel("claimed confidence interval")
    ax.set_ylabel("fraction containing the GALFIT value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    title = f"{split.capitalize()} coverage by {COMPARISON_KEYS.get(key, key)}"
    if best:
        title += f"\nbest: {best[2]} ({best[1]:.1f}% mean deviation)"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right",
              title=COMPARISON_KEYS.get(key, key), title_fontsize=7.5)
    return _save(fig, out_dir, f"{split}_coverage_sweep_{key}.png")


def compare_by(table, out_dir, key, split="devel"):
    """Accuracy and calibration grouped by one configuration axis.

    The same figure serves initialisation, freeze strategy, input geometry
    and pixel amplitude -- they differ only in which config key separates
    the runs, so writing four near-identical functions would just be four
    places for the metric definitions to drift apart.
    """
    df = table.dropna(subset=[key]) if key in table.columns else table.iloc[0:0]
    if df.empty or df[key].nunique() < 2:
        raise ValueError(f"runs do not differ in `{key}`")

    params = list(dict.fromkeys(df["parameter"]))
    groups = list(dict.fromkeys(df[key]))
    metrics = [("sigma_nmad", "residual $\\sigma$ (NMAD)"),
               ("coverage_dev", "mean coverage deviation [%]")]
    metrics = [(m, lab) for m, lab in metrics if m in df.columns]

    fig, axes = plt.subplots(
        len(metrics), len(params),
        figsize=(3.5 * len(params), 3.3 * len(metrics)), squeeze=False,
    )
    width = 0.8 / max(len(groups), 1)
    for row, (metric, mlabel) in enumerate(metrics):
        for c, param in enumerate(params):
            ax = axes[row, c]
            sub = df[df["parameter"] == param]
            for gi, g in enumerate(groups):
                vals = sub[sub[key] == g][metric].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                ax.bar(gi * width, np.nanmean(vals), width=width * 0.9,
                       label=str(g) if row == 0 and c == 0 else None)
            ax.set_xticks([])
            ax.set_ylabel(mlabel if c == 0 else "", fontsize=8)
            ax.set_title(pf._AXIS.get(param, (param, False))[0], fontsize=9)
            if metric == "coverage_dev":
                # G23 reports agreement within 4.5%; that is the bar to clear.
                ax.axhline(4.5, color="crimson", ls="--", lw=1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center",
               ncol=min(len(labels), 5), bbox_to_anchor=(0.5, 0.99),
               frameon=False, title=COMPARISON_KEYS.get(key, key),
               title_fontsize=8)
    fig.suptitle(f"{split.capitalize()}: comparison by "
                 f"{COMPARISON_KEYS.get(key, key)}", y=1.04)
    return _save(fig, out_dir, f"{split}_compare_{key}.png")


def learning_curve(table, out_dir, split="devel"):
    """Performance against the number of training galaxies.

    The question behind it is a budget question: whether another few
    thousand GALFIT fits would buy anything, or whether the model has
    already stopped improving and the compute is better spent elsewhere.
    """
    if "n_train" not in table.columns or table["n_train"].nunique() < 2:
        raise ValueError("runs do not differ in training-set size")
    params = list(dict.fromkeys(table["parameter"]))

    fig, axes = style.panel_grid(len(params), width=4.0, height=3.6)
    for ax, param in zip(axes, params):
        sub = table[table["parameter"] == param].sort_values("n_train")
        ax.plot(sub["n_train"], sub["sigma_nmad"], marker="o", lw=1.5,
                color="steelblue", label="residual $\\sigma$ (NMAD)")
        ax.set_xscale("log")
        ax.set_xlabel("training galaxies")
        ax.set_ylabel("residual $\\sigma$ (NMAD)")
        ax.set_title(pf._AXIS.get(param, (param, False))[0], fontsize=9)
        if "coverage_dev" in sub.columns:
            ax2 = ax.twinx()
            ax2.plot(sub["n_train"], sub["coverage_dev"], marker="s", ls=":",
                     lw=1.3, color="darkorange")
            ax2.set_ylabel("coverage deviation [%]", color="darkorange",
                           fontsize=8)
            ax2.tick_params(axis="y", labelcolor="darkorange", labelsize=7)
    fig.suptitle(f"{split.capitalize()}: learning curve", y=1.02)
    return _save(fig, out_dir, f"{split}_learning_curve.png")


# --- driver ------------------------------------------------------------------


def make_all(run_dirs, out_dir, split="devel"):
    runs = [load_run(d, split=split) for d in run_dirs]
    runs = [r for r in runs if r is not None]
    if not runs:
        raise SystemExit(
            f"no run had a posteriors_{split} cache; run "
            "`python -m ggt.visualization.posterior_figures <run_dir>` first"
        )
    log.info("comparing %d runs on the %s split", len(runs), split)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = check_comparable(runs)
    if notes:
        (out_dir / f"comparability_{split}.txt").write_text("\n".join(notes) + "\n")

    table = summary_table(runs, out_dir, split=split)

    made, failed = [], []

    def attempt(name, fn, *a, **kw):
        try:
            made.append(fn(*a, **kw))
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log.warning("%s: skipped (%s)", name, exc)

    for key in COMPARISON_KEYS:
        values = {json.dumps(r["cfg"].get(key)) for r in runs}
        if len(values) > 1:
            attempt(f"compare_{key}", compare_by, table, out_dir, key,
                    split=split)
    attempt("coverage_sweep", coverage_sweep, runs, out_dir, split=split)
    attempt("learning_curve", learning_curve, table, out_dir, split=split)

    log.info("comparison figures: %d written to %s", len(made), out_dir)
    if failed:
        log.info("comparison figures not applicable: %s", ", ".join(failed))
    return made


def main(argv=None):
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser(
        prog="python -m ggt.visualization.compare_runs",
        description="Compare finished runs on the same held-out split.",
    )
    ap.add_argument("out_dir")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--split", default="devel")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    make_all(args.run_dirs, args.out_dir, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
