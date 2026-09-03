#!/usr/bin/env python
"""Percentile coverage probabilities -- the calibration test.

The question this answers: of the galaxies whose predicted 68.27% interval
we drew, what fraction actually contain the GALFIT value? A well-calibrated
posterior puts that at 68.27%, and the curve of claimed-against-achieved
tracks the diagonal at every level.

This is what tunes `dropout_rate`, and it is the reason the whole MC-dropout
machinery exists: a mode with no trustworthy width is a point estimate
wearing a costume.  G22 §VI and Fig. 8, G23 Figs. 10 and 16.

Intervals are found by the same mode-outward walk as
`ggt.modules.result_aggregator.calculate_ci`, over the same KDE from
`calculate_cdf`, so a 68.27% interval means here exactly what it means
there. The difference is only that this module wants a whole curve of
percentiles and `calculate_ci` hardcodes three.
"""

from __future__ import annotations

import logging

import numpy as np

from ggt.modules.result_aggregator import calculate_cdf

log = logging.getLogger(__name__)

#: The claimed levels the curve is evaluated at. The three sigma levels are
#: included explicitly so the numbers quoted in text come off this same
#: curve rather than a second calculation that might drift from it.
DEFAULT_PERCENTILES = (
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.6827,
    0.75,
    0.85,
    0.9545,
    0.99,
    0.9973,
)


def mode_outward_intervals(samples, percentiles, n=1000, warnings=False):
    """Mode-centred credible intervals at every requested percentile.

    One KDE pass per galaxy serves the whole curve. Walking outward from
    the mode -- rather than taking symmetric quantiles -- is what makes
    these *highest-density* intervals, which is what GaMPEN reports and
    what makes them meaningful for a skewed posterior.

    Returns `(mode, {percentile: (lo, hi)})`.
    """
    x_domain, cdf, mode, y_pdf, mode_arg = calculate_cdf(
        samples, n=n, warnings=warnings
    )

    max_arg = len(x_domain) - 1
    upper_arg, lower_arg = mode_arg, mode_arg
    covered, lowers, uppers = [], [], []

    while (upper_arg < max_arg) or (lower_arg > 0):
        if upper_arg < max_arg:
            upper_arg += 1
        if lower_arg > 0:
            lower_arg -= 1
        covered.append(cdf[upper_arg] - cdf[lower_arg])
        lowers.append(lower_arg)
        uppers.append(upper_arg)

    covered = np.asarray(covered)
    out = {}
    for p in percentiles:
        k = int(np.abs(covered - p).argmin())
        out[p] = (float(x_domain[lowers[k]]), float(x_domain[uppers[k]]))
    return float(mode), out


def coverage_curve(samples, truths, percentiles=DEFAULT_PERCENTILES, n=1000):
    """Achieved coverage at each claimed percentile, for one parameter.

    `samples` is `(n_galaxies, n_draws)`; `truths` is `(n_galaxies,)`.
    Galaxies whose posterior is degenerate -- every draw identical, so the
    KDE has no width -- are dropped and counted rather than being silently
    scored as misses, because a collapsed posterior is a bug to chase, not
    a calibration result.

    Returns `(achieved, n_used, n_skipped)`.
    """
    samples = np.asarray(samples, dtype=float)
    truths = np.asarray(truths, dtype=float)
    percentiles = tuple(percentiles)

    hits = np.zeros(len(percentiles), dtype=float)
    n_used = n_skipped = 0

    for i in range(samples.shape[0]):
        draws = samples[i]
        draws = draws[np.isfinite(draws)]
        if draws.size < 2 or not np.isfinite(truths[i]) or draws.std() == 0:
            n_skipped += 1
            continue
        try:
            _, intervals = mode_outward_intervals(draws, percentiles, n=n)
        except Exception as exc:  # noqa: BLE001
            log.debug("galaxy %d: interval search failed (%s)", i, exc)
            n_skipped += 1
            continue
        n_used += 1
        for k, p in enumerate(percentiles):
            lo, hi = intervals[p]
            if lo <= truths[i] <= hi:
                hits[k] += 1.0

    if n_used == 0:
        raise ValueError("no galaxy had a usable posterior")
    if n_skipped:
        log.warning(
            "coverage: skipped %d/%d galaxies with degenerate posteriors",
            n_skipped,
            samples.shape[0],
        )
    return hits / n_used, n_used, n_skipped


def coverage_table(samples_by_target, truths_by_target, percentiles=DEFAULT_PERCENTILES, n=1000):
    """Coverage curves for every target, plus their mean.

    `samples_by_target` maps target name -> `(n_galaxies, n_draws)`;
    `truths_by_target` maps target name -> `(n_galaxies,)`.

    Returns `(percentiles, {target: achieved}, mean_achieved)`.
    """
    percentiles = tuple(percentiles)
    curves = {}
    for name, samples in samples_by_target.items():
        achieved, n_used, n_skipped = coverage_curve(
            samples, truths_by_target[name], percentiles=percentiles, n=n
        )
        curves[name] = achieved
        log.info(
            "coverage %-22s 68.27%% -> %.1f%%  (N = %d, skipped %d)",
            name,
            100 * achieved[list(percentiles).index(0.6827)]
            if 0.6827 in percentiles
            else np.nan,
            n_used,
            n_skipped,
        )
    mean = np.mean(np.vstack(list(curves.values())), axis=0)
    return np.asarray(percentiles), curves, mean


def deviation_from_diagonal(percentiles, achieved):
    """Mean absolute distance from perfect calibration, in percentage points.

    The single number to compare dropout rates on. G23 reports agreement
    within 4.5%; much worse than that means dropout is untuned or the model
    has not converged.
    """
    p = np.asarray(percentiles, dtype=float)
    a = np.asarray(achieved, dtype=float)
    return float(100.0 * np.mean(np.abs(a - p)))
