#!/usr/bin/env python
"""One matplotlib style, shared by every figure tier.

The papers' figures are visually consistent and ours should be too: the
per-run diagnostics (`training_figures`), the posterior figures
(`posterior_figures`) and the cross-run comparison (`compare_runs`) all draw
through the helpers here, so a panel means the same thing wherever it turns
up.

`display_space` is the one that matters for reading a plot correctly. The
network trains on `custom_logit_bt` because a logit is unbounded and
symmetric, which suits a Gaussian likelihood -- but bulge-to-total is a
fraction, and a plot of it should run 0 to 1.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from pathlib import Path  # noqa: E402

import logging  # noqa: E402

log = logging.getLogger(__name__)

DPI = 130

#: Colour floor and draw threshold for every population 2D histogram. Both
#: are 1: a cell holding a single galaxy is drawn, and the colour scale
#: starts there. Raising them thins the sparse wings, which reads as
#: cleaner but drops the isolated outliers these diagnostics exist to
#: show -- tried at 3 and reverted.
HEXBIN_VMIN = 1
HEXBIN_MINCNT = 1

#: Type sizes for the coverage figures, which live in two different modules
#: (`posterior_figures.coverage_figure` and `compare_runs.coverage_sweep`)
#: and would otherwise drift apart.
COVERAGE_FONTS = {"title": 12, "label": 12.5, "tick": 11, "legend": 10}

#: Targets whose residuals are best judged as a fraction of the truth.
#: `custom_logit_bt` is deliberately absent: B/T is already a fraction, and
#: dividing a fraction by a fraction near zero explodes.
FRACTIONAL_TARGETS = ("ln_R_e_asec", "ln_total_flux_adus")


def save(fig, out_dir, name):
    """Tight-layout, write at the shared DPI, close, and log the path."""
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def pretty(name):
    return name.replace("_", " ")


def display_space(name, values):
    """Map a target into the units it is easiest to judge in.

    `expit` is the plain inverse of the logit; `logit_custom` additionally
    nudges exact 0s and 1s inward by an epsilon derived from the training
    column, so values that started at exactly 0 or 1 come back very close
    to, but not exactly at, the endpoints. That is immaterial for a
    diagnostic plot -- and in our Euclid GALFIT labels no `bt` is exactly 0
    or 1 anyway, so the epsilon branch never fires.

    Returns `(values, label, limits)`; limits is None to autoscale.
    """
    if name == "custom_logit_bt":
        from scipy.special import expit

        return expit(values), "B/T", (0.0, 1.0)
    return values, pretty(name), None


def residual(name, pred, true):
    """Residual in the units the parameter is judged in.

    For the two log targets the difference in log space *is* the fractional
    residual once exponentiated: ``ln(p) - ln(t) = ln(p/t)``, so
    ``exp(delta) - 1`` is exactly ``(p - t) / t`` with no approximation.
    That is what G22 Fig. 16 plots, and it is the scale-free quantity you
    want when comparing a 0.2" galaxy against a 3" one.

    For B/T the plain difference is used instead. B/T is already a
    fraction, and dividing by a truth that runs down to 0.007 in our sample
    would manufacture enormous residuals out of small absolute errors.

    Returns ``(values, label)``.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if name in FRACTIONAL_TARGETS:
        with np.errstate(over="ignore"):
            return np.exp(pred - true) - 1.0, f"({pretty(name)} frac. residual)"
    if name == "custom_logit_bt":
        from scipy.special import expit

        return expit(pred) - expit(true), "B/T residual"
    return pred - true, f"{pretty(name)} residual"


def hexbin(
    ax, x, y, gridsize=35, extent=None, cmap="viridis", colorbar=True,
    xscale=None, vmin=HEXBIN_VMIN, mincnt=HEXBIN_MINCNT,
):
    """Hexbin on a log colour scale -- the papers' convention.

    A linear scale on a galaxy scatter is dominated by the one dense clump
    at the centre of the distribution and shows nothing about the wings,
    which is exactly where a model fails.

    `mincnt=1` keeps single-galaxy cells visible rather than blending them
    into the background, and `vmin` starts the colour scale at that same
    count. The wings are where a model fails, so an isolated galaxy is
    signal here, not noise to be tidied away.

    Both are parameters should a particular panel want a higher floor, but
    note that raising `mincnt` hides points which `overlay_percentiles` and
    `robust_ylim` still include -- they are computed from every finite
    point, so the dotted percentile curves would then run outside the drawn
    hexes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        raise ValueError("no finite points to plot")

    # `xscale` must be handed to hexbin, not set on the axis afterwards.
    # hexbin lays its grid out in data coordinates, so an axis switched to
    # log after the fact leaves every cell piled into the left-hand sliver
    # and stretched into stripes. Passing it here bins in log space, which
    # is what was meant.
    kw = {} if xscale is None else {"xscale": xscale}
    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        norm=LogNorm(vmin=vmin),
        extent=extent,
        **kw,
    )
    if colorbar:
        cb = ax.figure.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("count", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return hb


def robust_ylim(ax, values, q=(0.5, 99.5), pad=0.08):
    """Clip the y-axis to the bulk of the distribution.

    A fractional residual has no upper bound, and a single galaxy whose
    size is over-predicted eightfold will otherwise compress fourteen
    hundred well-behaved ones into a single row of pixels. The count of
    what falls outside is written on the axis so the clipping is never
    silent.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = np.percentile(v, q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    span = hi - lo
    lo, hi = lo - pad * span, hi + pad * span
    n_out = int(((v < lo) | (v > hi)).sum())
    ax.set_ylim(lo, hi)
    if n_out:
        ax.text(
            0.02,
            0.98,
            f"{n_out} outside",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="0.35",
        )


def identity_line(ax, lo, hi, color="w", ls="--", lw=1.2):
    """The `y = x` line every predicted-vs-true panel carries."""
    ax.plot([lo, hi], [lo, hi], color=color, ls=ls, lw=lw, zorder=5)


def zero_line(ax, color="k", ls="--", lw=1.0):
    """The `y = 0` line every residual panel carries."""
    ax.axhline(0.0, color=color, ls=ls, lw=lw, zorder=5)


def stats_box(ax, values, loc=(0.03, 0.97), fontsize=8):
    """The boxed mu / median / sigma of G22 Fig. 12 and G23 Fig. 18."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    ax.text(
        loc[0],
        loc[1],
        f"$\\mu$ {v.mean():+.4f}\nmed {np.median(v):+.4f}\n"
        f"$\\sigma$ {v.std():.4f}\nN {v.size}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )


def running_percentiles(x, y, bins=12, qs=(16, 50, 84)):
    """Percentiles of `y` in bins of `x`, for the trend line on a hexbin.

    Returns `(centres, {q: values})`. Bins with fewer than five points are
    dropped rather than plotted: a 16th percentile from three galaxies is
    noise drawn with the authority of a line.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.array([]), {q: np.array([]) for q in qs}

    edges = np.percentile(x, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    centres, out = [], {q: [] for q in qs}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 5:
            continue
        centres.append(0.5 * (lo + hi))
        for q in qs:
            out[q].append(np.percentile(y[sel], q))
    return np.asarray(centres), {q: np.asarray(v) for q, v in out.items()}


def overlay_percentiles(ax, x, y, bins=12, color="crimson"):
    """Draw the 16/50/84 trend of `y` vs `x` over a hexbin panel."""
    centres, qs = running_percentiles(x, y, bins=bins)
    if centres.size == 0:
        return
    ax.plot(centres, qs[50], color=color, lw=1.6, zorder=6, label="median")
    ax.plot(centres, qs[16], color=color, lw=1.0, ls=":", zorder=6)
    ax.plot(centres, qs[84], color=color, lw=1.0, ls=":", zorder=6, label="16/84%")


def panel_grid(n, width=4.3, height=4.0, squeeze=True):
    """One row of `n` panels at the shared aspect ratio."""
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return (fig, axes[0]) if squeeze else (fig, axes)
