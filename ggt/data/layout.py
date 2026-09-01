# -*- coding: utf-8 -*-
"""Where a (z_bin, band) dataset and its run outputs live on disk.

One place that knows the directory convention, so bash, python and the
notebooks cannot drift apart. Nothing here touches the filesystem except
`ensure`.

Data layout, under `data_root()`::

    <root>/gampen_data/
      z0_0.00_0.25/
        cutouts.npy        (N, 4, crop, crop) float32 -- ALL sources in the
                           bin, ALL 4 bands
        object_ids.npy     (N,) int64 -- row order of cutouts.npy
        cache_manifest.json
        VIS/
          info.csv         labels + provenance, AFTER this band's cuts
          splits/euclid-<seed>-{train,devel,test}.csv
          splits/euclid-<seed>-manifest.json
          splits/euclid-<seed>-scaler.joblib

The pixel cache is shared by all four bands and is deliberately
*quality-cut agnostic*: cuts live only in each band's `info.csv`, so
retuning a cut rewrites a small CSV instead of rebuilding tens of GB.

**Outputs are split by size, not by kind.** A `best.pt`/`last.pt` pair is
1.2 GB at cutout_size=400, so weights, metrics and logs go to the data
volume beside the cache; figures are small and go in the analysis repo
where you can actually look at them. Neither ever lands inside this
package -- that would put gigabytes of checkpoints inside a git submodule.
"""

from __future__ import annotations

import os
from pathlib import Path

from ggt.surveys import euclid

DEFAULT_DATA_ROOT = "/astro/store/shire/cgwill/Euclid_Gampen_Data"


def data_root() -> Path:
    """Root of the off-repo data tree.

    Override with `EUCLID_GAMPEN_DATA_ROOT`.
    """
    return Path(os.environ.get("EUCLID_GAMPEN_DATA_ROOT", DEFAULT_DATA_ROOT))


def gampen_data(root=None) -> Path:
    return Path(root or data_root()) / "gampen_data"


def bin_dir(z_bin: int, root=None) -> Path:
    return gampen_data(root) / euclid.bin_dirname(z_bin)


def cache_path(z_bin: int, root=None) -> Path:
    return bin_dir(z_bin, root) / "cutouts.npy"


def object_ids_path(z_bin: int, root=None) -> Path:
    return bin_dir(z_bin, root) / "object_ids.npy"


def cache_manifest_path(z_bin: int, root=None) -> Path:
    return bin_dir(z_bin, root) / "cache_manifest.json"


def band_dir(z_bin: int, band: str, root=None) -> Path:
    if band not in euclid.BAND_INDEX:
        raise ValueError(
            f"unknown band {band!r}; expected one of {euclid.BANDS}"
        )
    return bin_dir(z_bin, root) / band


def info_csv(z_bin: int, band: str, root=None) -> Path:
    return band_dir(z_bin, band, root) / "info.csv"


def splits_dir(z_bin: int, band: str, root=None) -> Path:
    return band_dir(z_bin, band, root) / "splits"


def split_manifest(z_bin: int, band: str, slug: str, root=None) -> Path:
    """Per-slug, so seeds coexist and each stays independently verifiable.

    A single shared manifest would be overwritten by the next
    `--force-resplit`, silently making every earlier seed's split
    unverifiable.
    """
    return splits_dir(z_bin, band, root) / f"{slug}-manifest.json"


def scaler_path(z_bin: int, band: str, slug: str, root=None) -> Path:
    """Per-slug too -- a scaler is fitted on one specific train split."""
    return splits_dir(z_bin, band, root) / f"{slug}-scaler.joblib"


def pretrained_dir(root=None) -> Path:
    return Path(root or data_root()) / "gampen_pretrained"


# --- run outputs -------------------------------------------------------


def runs_root(root=None) -> Path:
    """Where checkpoints and metrics go. Big; lives on the data volume."""
    override = os.environ.get("EUCLID_GAMPEN_RUNS_ROOT")
    if override:
        return Path(override)
    return Path(root or data_root()) / "gampen_runs"


def logs_root(root=None) -> Path:
    override = os.environ.get("EUCLID_GAMPEN_LOGS_ROOT")
    if override:
        return Path(override)
    return Path(root or data_root()) / "gampen_logs"


def run_dir(z_bin: int, band: str, run_name: str, root=None) -> Path:
    return runs_root(root) / euclid.bin_dirname(z_bin) / band / run_name


def log_dir(z_bin: int, band: str, root=None) -> Path:
    return logs_root(root) / euclid.bin_dirname(z_bin) / band


def figures_root() -> Path:
    """Where figures go: small, and wanted in the analysis repo.

    This package must not hardcode a path into whatever repo contains it,
    so resolution is: `EUCLID_GAMPEN_FIGURES_ROOT`, else `<cwd>/figures`.
    The launcher scripts know their repo root and export the variable, so
    from `train_zbin.sh` or `make_figs.sh` this is automatic. Callers may
    also pass an explicit directory and bypass this entirely.
    """
    override = os.environ.get("EUCLID_GAMPEN_FIGURES_ROOT")
    if override:
        return Path(override)
    return Path.cwd() / "figures"


def figures_dir(run_name: str) -> Path:
    return figures_root() / "training" / run_name


def ensure(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# --- shell interface ---------------------------------------------------


def main(argv=None) -> int:
    """Print shell assignments so bash and python cannot drift apart.

        eval "$(python -m ggt.data.layout --z-bin 0 --band VIS \
                    --run-name my_run)"

    defines DATA_ROOT, BIN_DIR, BAND_DIR, INFO_CSV, CACHE_NPY, RUN_DIR,
    LOG_DIR and FIGURES_DIR. The launchers use this rather than rebuilding
    the directory convention in bash, where it would immediately fall out
    of step.
    """
    import argparse

    p = argparse.ArgumentParser(description="Print dataset/run paths.")
    p.add_argument("--z-bin", type=int, required=True)
    p.add_argument("--band", required=True, choices=euclid.BANDS)
    p.add_argument("--run-name", default="run")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)

    root = args.data_root
    values = {
        "DATA_ROOT": data_root() if root is None else Path(root),
        "BIN_DIR": bin_dir(args.z_bin, root),
        "BAND_DIR": band_dir(args.z_bin, args.band, root),
        "INFO_CSV": info_csv(args.z_bin, args.band, root),
        "CACHE_NPY": cache_path(args.z_bin, root),
        "RUN_DIR": run_dir(args.z_bin, args.band, args.run_name, root),
        "LOG_DIR": log_dir(args.z_bin, args.band, root),
        "FIGURES_DIR": figures_dir(args.run_name),
    }
    for key, value in values.items():
        print(f'{key}="{value}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
