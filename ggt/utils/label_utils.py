# -*- coding: utf-8 -*-
"""Label transforms: the forward direction of what `unscale_preds` undoes.

Two things here are easy to get wrong and expensive to get wrong quietly.

1. **The logit transform must match the inverse exactly.** `logit_custom`
   is copied verbatim from the GaMPEN docs.
   `result_aggregator.expit_custom` inverts it *using the training `bt`
   column* to re-derive the epsilons it used for exact 0s and 1s -- so if
   the forward transform differs at the extremes, the inverse is silently
   wrong there.

2. **The Pogson constant.** We use 2.5. The published HSC pipeline uses
   2.512, a self-consistent but nonstandard *definition* inherited from the
   Yale notebooks -- verified against a real row of
   `g_0_025/scaling_data_dir/info.csv`: `d_mag=19.1324` ->
   `d_flux_adus=1355.2093`, which is `10**((27-19.1324)/2.512)` exactly.
   Because both directions use it, it cancels and their catalog is
   internally coherent (residual vs a true AB magnitude is only
   -0.0028 mag). **The hazard is mixing the two**: our forward transform
   with their hardcoded inverse puts a 0.48% brightness-dependent slope
   into `total_mag` (-0.014 mag at m=24, -0.053 at m=16). `unscale_preds`
   takes `pogson=`; pass 2.512 ONLY for the HSC zero-shot baseline.

The zeropoints and the `uJy -> ADU` factor are survey facts and live in
`ggt.surveys.euclid`; they are accepted here as arguments.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logit

from ggt.surveys.euclid import HSC_ZP, POGSON, UJY_TO_ADU


def logit_custom(x_input):
    """Logit with handling for exact 0s and 1s. Verbatim from the docs.

    Takes the ENTIRE column, not a single value: the epsilons depend on the
    column's own extremes, which is also how `expit_custom` reconstructs
    them.
    """
    x = np.array(x_input, dtype=float)

    if np.min(x) < 0 or np.max(x) > 1:
        raise ValueError("x must be between 0 and 1")

    if np.min(x) == 0:
        min_x = np.min(x[x != 0])
        add_epsilon = min_x / 2.0
        x[np.where(x == 0)[0]] = add_epsilon

    if np.max(x) == 1:
        max_x = np.max(x[x != 1])
        sub_epsilon = (1 - max_x) / 2.0
        x[np.where(x == 1)[0]] = 1.0 - sub_epsilon

    return logit(x)


def ujy_to_adu(flux_ujy, factor=UJY_TO_ADU):
    """uJy -> ADU on HSC's ZP 27."""
    return np.asarray(flux_ujy, dtype=float) * float(factor)


def adu_to_ujy(flux_adu, factor=UJY_TO_ADU):
    return np.asarray(flux_adu, dtype=float) / float(factor)


def flux_to_mag(flux, zp=HSC_ZP, pogson=POGSON):
    """`mag = -pogson * log10(flux) + zp`.

    Defaults give the standard relation on HSC's ZP. Pass `pogson=2.512`
    only to reproduce the published HSC catalogs' own convention.
    """
    f = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(f > 0, f, np.nan)
        return -float(pogson) * np.log10(safe) + float(zp)


def mag_to_flux(mag, zp=HSC_ZP, pogson=POGSON):
    m = np.asarray(mag, dtype=float)
    return 10.0 ** ((float(zp) - m) / float(pogson))


def build_targets(bt, re_asec, flux_ujy):
    """The three GaMPEN targets, in the pretrained head's order.

    Returns a dict with `custom_logit_bt`, `ln_R_e_asec`,
    `ln_total_flux_adus` plus the physical columns they derive from.
    **Never permute the target order** -- it is the order of the pretrained
    output layer.
    """
    bt = np.asarray(bt, dtype=float)
    re_asec = np.asarray(re_asec, dtype=float)
    flux_ujy = np.asarray(flux_ujy, dtype=float)

    flux_adu = ujy_to_adu(flux_ujy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_re = np.log(np.where(re_asec > 0, re_asec, np.nan))
        ln_flux_adu = np.log(np.where(flux_adu > 0, flux_adu, np.nan))
        ln_flux_ujy = np.log(np.where(flux_ujy > 0, flux_ujy, np.nan))

    return {
        "bt": bt,
        "R_e_asec": re_asec,
        "total_flux_ujy": flux_ujy,
        "total_flux_adus": flux_adu,
        "total_mag": flux_to_mag(flux_adu),
        "custom_logit_bt": logit_custom(bt),
        "ln_R_e_asec": ln_re,
        "ln_total_flux_adus": ln_flux_adu,
        "ln_total_flux_ujy": ln_flux_ujy,
    }
