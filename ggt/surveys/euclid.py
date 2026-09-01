# -*- coding: utf-8 -*-
"""Euclid constants, geometry, and the map onto the published HSC models.

This is the single place in `ggt/` that knows anything Euclid-specific.
Swapping in another survey means writing a sibling module with the same
names, not editing the data/model/training code.

Values here are the *authoritative* ones, cross-checked against three
sources:

  * the selection config that `zbin_maguniform` round-trips into the
    production subset's FITS header (`REDSHIFT_BINS`, `TARGET_CROP_AS`) --
    read it with `config_from_catalog` rather than trusting the defaults
    below when a catalog is to hand;
  * `ToDo.txt` in the Euclid_GAMPEN repo, which records the cutout sizes;
  * `pipeline/common/euclid_dataset.py` there, for the pixel zeropoints.
"""

from __future__ import annotations

import json
import re

import numpy as np

# --- geometry ---------------------------------------------------------
# Every band sits on the MER 0.1"/px grid: NISP is resampled onto the VIS
# grid, so there is a single plate scale for all four. (The 0.3"/px figure
# quoted for native NISP does NOT apply to these data products.)
PLATE_SCALE = 0.1  # arcsec / pixel, all bands
ON_DISK_PX = 400  # every science shard stamp is a flat 400 px / 40"

# Redshift bin edges -> 6 bins, indexed 0..5.
REDSHIFT_BINS = [0, 0.25, 0.5, 0.75, 1.25, 1.75, 3]

# Intended cutout side per bin, in arcsec. Bin 0 wants 46" but only 40"
# exists on disk, so target_crop_px caps at ON_DISK_PX.
TARGET_CROP_AS = [46, 32, 24, 20, 16, 16]

# --- bands ------------------------------------------------------------
# Canonical (hyphenated) order, matching the shard channel order.
BANDS = ["VIS", "NIR-Y", "NIR-J", "NIR-H"]
BAND_INDEX = {b: i for i, b in enumerate(BANDS)}

# Token used in galfit_results.fits columns, e.g. `vis_bt`, `nir_h_re_tot`.
BAND_TOKEN = {
    "VIS": "vis",
    "NIR-Y": "nir_y",
    "NIR-J": "nir_j",
    "NIR-H": "nir_h",
}

# Short token used by the selection code's `sel_band` column.
BAND_FROM_SEL = {"VIS": "VIS", "Y": "NIR-Y", "J": "NIR-J", "H": "NIR-H"}

# `sel_band` per bin, matching NORM_BINS in the production selection config:
# the bluer bins are selected and magnitude-normalised in VIS, the redder
# ones in Y/J/H where their sources live.
SEL_BAND_PER_BIN = ["VIS", "VIS", "VIS", "NIR-Y", "NIR-J", "NIR-H"]

# --- photometry -------------------------------------------------------
# THE TWO ZEROPOINTS:
#   catalog fluxes in uJy  -> 23.9
#   IMAGE PIXELS           -> per-band MAGZERO below
# Crossing them is ~0.7 mag wrong in VIS and ~6 mag in NIR.
MER_AB_ZP = 23.9
MAGZERO = {"VIS": 24.6, "NIR-Y": 29.8, "NIR-J": 30.0, "NIR-H": 29.9}

# HSC's ADU zeropoint. Our labels are expressed on this scale so that
# `unscale_preds` and `total_mag` stay self-consistent.
HSC_ZP = 27.0

# Pogson constant. 2.5 is correct. The published HSC pipeline uses 2.512 --
# a self-consistent but nonstandard *definition*, not a bug (it cancels,
# because both directions use it). Never mix the two.
POGSON = 2.5
HSC_POGSON = 2.512

# uJy -> "HSC-equivalent ADU" on ZP 27. ln of this is what GaMPEN trains on.
UJY_TO_ADU = float(10.0 ** ((HSC_ZP - MER_AB_ZP) / 2.5))  # 17.3780...

# The three targets, in the pretrained output head's order.
# NEVER permute: the order is baked into the HSC checkpoints.
TARGET_COLUMNS = ("custom_logit_bt", "ln_R_e_asec", "ln_total_flux_adus")

# --- the published HSC checkpoints ------------------------------------
MODEL_TYPE = "vgg16_w_stn_oc_drp"

# z_bin -> (real-data checkpoint dir, simulated checkpoint dir).
#
# The mapping is by **redshift**, not angular size, because our bins 0/1/2
# coincide exactly with GaMPEN's low/mid/high-z bins -- so the rest-frame
# band choice carries over too, not just the redshift range. Bins 3-5 have
# no counterpart and take the highest-z model available.
CHECKPOINT_FOR_BIN = {
    0: ("g_0_025", "sim_g_0_025"),  # z<0.25   <-> HSC low-z  g
    1: ("r_025_050", "sim_r_025_050"),  # 0.25-0.5 <-> HSC mid-z  r
    2: ("i_050_075", "sim_i_050_075"),  # 0.5-0.75 <-> HSC high-z i
    3: ("i_050_075", "sim_i_050_075"),  # no HSC counterpart above z=0.75:
    4: ("i_050_075", "sim_i_050_075"),  # use the highest-z model available
    5: ("i_050_075", "sim_i_050_075"),
}

# Published filenames. The names in GaMPEN's docs are WRONG (they 404), and
# the three simulated models use three different conventions -- do not
# pattern-generate these.
CHECKPOINT_FILE = {
    "g_0_025": "g_0_025_real_data.pt",
    "r_025_050": "r_025_050_real_data.pt",
    "i_050_075": "i_050_075_real_data.pt",
    "sim_g_0_025": "sim_g_0_025.pt",
    "sim_r_025_050": "sim_r_025_050_model.pt",
    "sim_i_050_075": "sim_i_050_075.pt",
}

# Published dropout rates, by checkpoint. Ghosh et al. tuned these for
# calibrated coverage; they are the right starting point but must be
# re-tuned for Euclid.
DROPOUT_FOR = {
    "g_0_025": 4e-4,
    "r_025_050": 2e-4,
    "i_050_075": 2e-4,
    "sim_g_0_025": 7e-4,
    "sim_r_025_050": 7e-4,
    "sim_i_050_075": 4e-4,
}


def n_bins() -> int:
    return len(REDSHIFT_BINS) - 1


def target_crop_px(
    z_bin: int,
    plate_scale: float = PLATE_SCALE,
    on_disk_px: int = ON_DISK_PX,
) -> int:
    """Intended cutout side in px for a bin, capped at what is on disk.

    Mirrors `zbin_maguniform._target_crop_px` exactly; kept in step with it
    deliberately.
    """
    return int(min(round(TARGET_CROP_AS[z_bin] / plate_scale), on_disk_px))


def bin_dirname(z_bin: int) -> str:
    """Self-describing directory name, e.g. `z0_0.00_0.25`."""
    lo = REDSHIFT_BINS[z_bin]
    hi = REDSHIFT_BINS[z_bin + 1]
    return f"z{z_bin}_{lo:.2f}_{hi:.2f}"


def zbin_of(redshift):
    """Map redshift(s) to a bin index, or -1 when outside the grid.

    NOT used by the production path -- the dataset builders require a real
    `z_bin` column and fail loudly without one. This exists to *construct*
    test fixtures from catalogs that predate the column.

    Half-open bins `[lo, hi)`, matching `np.digitize` semantics in the
    selection code.
    """
    z = np.asarray(redshift, dtype=float)
    edges = np.asarray(REDSHIFT_BINS, dtype=float)
    out = np.digitize(z, edges) - 1
    inside = np.isfinite(z) & (z >= edges[0]) & (z < edges[-1])
    return np.where(inside, out, -1).astype(int)


def pixel_zp_scale(band: str, pixel_zp) -> float:
    """Factor to move pixels onto `pixel_zp`; 1.0 when disabled.

    `pixel_zp=None` (the default everywhere) leaves pixels in native MER
    units. `27` puts them on HSC's ADU scale, which is what the arsinh
    normalisation was tuned against. The decision was to leave this OFF and
    let fine-tuning absorb the ~9x amplitude difference; reach for it only
    if training misbehaves.
    """
    if pixel_zp is None:
        return 1.0
    return float(10.0 ** ((float(pixel_zp) - MAGZERO[band]) / 2.5))


def config_from_catalog(path):
    """Recover the selection config JSON from a subset catalog's header.

    Returns `None` for catalogs that predate it (e.g. a debug subset).
    """
    from astropy.io import fits

    header = fits.getheader(str(path), 1)
    if "COMMENT" not in header:
        return None
    blob = "".join(str(c) for c in header["COMMENT"])
    match = re.search(r"\{.*\}", blob, re.S)
    if not match:
        return None
    try:
        cfg = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return cfg.get("zbin_maguniform", cfg)
