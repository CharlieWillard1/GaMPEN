# -*- coding: utf-8 -*-
"""Build a `FITSDataset` backed by a per-z-bin contiguous pixel cache.

There are no per-source FITS files. Pixels come from `cutouts.npy` -- one
contiguous array per z-bin holding all four bands -- and are handed to
`FITSDataset` through its `observations=` argument. Everything downstream
of that point is stock: arsinh normalisation, kornia transforms,
`repeat_dims`, `expand_factor`.

`crop_px` (what the cache holds) and `cutout_size` (what the network sees)
are separate. The cache is built at the bin's `target_crop_px`, so a run
can crop *smaller* but not larger.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ggt.data import layout
from ggt.surveys import euclid
from ggt.data.crop import crop_center

log = logging.getLogger("ggt.data.cache_dataset")


def load_catalog(
    z_bin, band, split=None, slug=None, root=None
) -> pd.DataFrame:
    """``info.csv`` for the whole band, or one split of it."""
    bd = layout.band_dir(z_bin, band, root)
    path = bd / "splits" / f"{slug}-{split}.csv" if split else bd / "info.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it with:\n"
            f"    python -m euclid_prep.build_info --z-bin {z_bin} "
            f"--band {band} "
            f"--subset-catalog <fits>"
            + ("\n    python -m ggt.data.splits ..." if split else "")
        )
    return pd.read_csv(path)


def load_observations(z_bin, band, cache_rows, cutout_size=None, root=None):
    """Pull one band's pixels for the given cache rows out of ``cutouts.npy``.

    Returns `(list_of_2d_arrays, crop_px_used)`. Reading a memmap with a
    fancy index materialises the selection in RAM, which is what we want:
    `FITSDataset` holds every image in memory regardless, so this is the
    same footprint the FITS path would have had.
    """
    manifest = json.loads(layout.cache_manifest_path(z_bin, root).read_text())
    crop_px = int(manifest["crop_px"])
    band_idx = euclid.BAND_INDEX[band]

    arr = np.load(layout.cache_path(z_bin, root), mmap_mode="r")
    rows = np.asarray(cache_rows, dtype=int)
    if rows.min() < 0 or rows.max() >= arr.shape[0]:
        raise SystemExit(
            f"cache_row out of range: got [{rows.min()}, "
            f"{rows.max()}] for a cache with {arr.shape[0]} rows. The "
            f"info.csv and the cache are out of sync -- rebuild "
            f"info.csv after rebuilding the cache."
        )

    log.info(
        "reading %d x %s from %s (band %s, idx %d)",
        len(rows),
        (crop_px, crop_px),
        layout.cache_path(z_bin, root).name,
        band,
        band_idx,
    )
    obs = np.asarray(arr[rows, band_idx])  # (n, crop, crop)

    if cutout_size is not None and cutout_size != crop_px:
        if cutout_size > crop_px:
            raise SystemExit(
                f"cutout_size={cutout_size} exceeds the cache's "
                f"crop_px={crop_px}. The cache is "
                f"built at the bin's target_crop_px; rebuild it larger with "
                f"`build_cache --crop-px {cutout_size}` if you really "
                f"want a wider field."
            )
        log.info("cropping %d -> %d px on load", crop_px, cutout_size)
        obs = crop_center(obs, cutout_size, crop_px)
        crop_px = cutout_size

    n_bad = int((~np.isfinite(obs)).any(axis=(1, 2)).sum())
    if n_bad:
        raise SystemExit(
            f"{n_bad} of {len(obs)} images contain non-finite pixels; "
            "a NaN would propagate silently into the aleatoric loss"
        )
    return [np.ascontiguousarray(o) for o in obs], crop_px


def make_dataset(
    z_bin,
    band,
    split=None,
    slug=None,
    root=None,
    target_metrics=None,
    cutout_size=None,
    channels=3,
    repeat_dims=True,
    normalize=True,
    transform=None,
    expand_factor=1,
    scaler=None,
    label_scaling=None,
):
    """A stock `FITSDataset` whose pixels come from the cache.

    `scaler` (a fitted sklearn transformer) is applied to the labels here
    rather than letting `standardize_labels` refit one per construction:
    refitting is deterministic given a fixed split, so it is not wrong, but
    it is invisible and unverifiable.
    """
    from ggt.data import FITSDataset

    target_metrics = list(target_metrics or config_target_metrics())
    df = load_catalog(z_bin, band, split, slug, root)
    missing = [
        c
        for c in target_metrics + ["cache_row", "file_name"]
        if c not in df.columns
    ]
    if missing:
        raise SystemExit(f"info.csv is missing {missing}")

    obs, crop_px = load_observations(
        z_bin, band, df["cache_row"].to_numpy(), cutout_size, root
    )
    if cutout_size is None:
        cutout_size = crop_px

    ds = FITSDataset(
        data_dir=layout.band_dir(z_bin, band, root),
        slug=slug,
        split=split,
        cutout_size=cutout_size,
        channels=channels,
        label_col=target_metrics,
        normalize=normalize,
        transform=transform,
        expand_factor=expand_factor,
        repeat_dims=repeat_dims,
        label_scaling=label_scaling,  # None: we scale explicitly below
        observations=obs,
    )
    if scaler is not None:
        ds.labels = scaler.transform(
            np.asarray(df[target_metrics], dtype=float)
        )
    return ds


def config_target_metrics():
    """The survey's target columns, in the pretrained head's order."""
    return euclid.TARGET_COLUMNS
