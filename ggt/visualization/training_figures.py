#!/usr/bin/env python
"""Diagnostic figures for a finished (or running) stage-04 training run.

    python -m ggt.visualization.training_figures <run_dir>

Reads only ``metrics.csv``, ``config.json``, ``best.pt`` and the devel split,
so it re-runs standalone against any run directory.

Every figure is attempted independently: one that cannot be made -- because
the run died before writing a checkpoint, say -- logs a warning and the rest
are still produced.

The two Euclid-specific diagnostics are ``stn_scale_hist`` and
``input_pixel_hist``. They carry more weight than usual because we do not
rescale pixels by default, and the two things most likely to go wrong in this
transfer are the re-initialised STN head collapsing and the input dynamic
range sitting in the wrong regime -- neither of which shows up clearly in a
loss curve.
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

from ggt.data import cache_dataset, layout, splits  # noqa: E402

from ggt.visualization import style  # noqa: E402

log = logging.getLogger(__name__)

# The style module owns these now, so the three figure tiers cannot drift
# apart. Kept as module-level names because callers elsewhere import them
# from here.
DPI = style.DPI
_save = style.save
_pretty = style.pretty
display_space = style.display_space


# --- shared helpers ----------------------------------------------------------


def _targets(cfg):
    return list(cfg.get("target_columns") or [])


def rebuild_devel(cfg, split="devel"):
    """Rebuild one split's dataset exactly as the run built it."""
    root = cfg.get("data_root")
    # The scaler is always fitted on the survey's full target list, even
    # when a run trained on a subset of it, so load it with that list and
    # then narrow it. Asking for the subset directly fails the column
    # check and takes the whole figure pass down with it.
    full = list(cfg.get("survey_target_columns") or _targets(cfg))
    scaler = splits.load_scaler(
        cfg["z_bin"],
        cfg["band"],
        seed=cfg["seed"],
        root=root,
        target_metrics=full,
    )
    if _targets(cfg) != full:
        idx = [full.index(t) for t in _targets(cfg)]
        scaler = splits.ColumnSubsetScaler(scaler, idx)

    import kornia.augmentation as K
    import torch.nn as nn

    cutout = cfg["resolved_cutout_size"]
    ds = cache_dataset.make_dataset(
        cfg["z_bin"],
        cfg["band"],
        split=split,
        slug=splits.slug_for(cfg["seed"]),
        root=root,
        target_metrics=_targets(cfg),
        cutout_size=cutout,
        channels=cfg.get("channels", 3),
        repeat_dims=cfg.get("repeat_dims", True),
        normalize=cfg.get("normalize", True),
        transform=nn.Sequential(K.CenterCrop(cutout)),
        expand_factor=1,
        scaler=scaler,
    )
    if cfg.get("limit"):
        n = min(cfg["limit"], len(ds.labels))
        ds.labels = ds.labels[:n]
        ds.observations = ds.observations[:n]
        ds.filenames = ds.filenames[:n]
        ds.data_info = ds.data_info.iloc[:n]
    return ds, scaler


def load_best_model(cfg, run_dir):
    """Rebuild the architecture and load ``best.pt`` into it."""
    import torch

    from ggt.models import transfer as models

    ckpt = Path(run_dir) / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no best.pt in {run_dir}")

    net = models.build_model(
        cutout_size=cfg["resolved_cutout_size"],
        n_out=cfg.get("n_out", 9),
        channels=cfg.get("channels", 3),
        dropout_rate=cfg.get("dropout"),
        parallel=False,
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    from ggt.utils import load_checkpoint_tolerant

    load_checkpoint_tolerant(net, state, allow_reinit=set())
    net.eval()
    return net


def _predict(net, ds, batch_size=16, max_n=4000, device=None):
    """Mean predictions (first `n_target` outputs) and truths, both scaled.

    `device` lets this run against the live model mid-training, which sits
    on the GPU; the standalone path leaves it None and works on CPU.
    """
    import torch

    n = min(len(ds.labels), max_n)
    preds, truths = [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = torch.stack([ds[i][0] for i in range(start, stop)])
            if device is not None:
                batch = batch.to(device)
            out = net(batch)
            preds.append(out[:, : ds.labels.shape[1]].cpu().numpy())
            truths.append(np.asarray(ds.labels[start:stop], dtype=float))
    return np.concatenate(preds), np.concatenate(truths)


# --- metrics.csv figures -----------------------------------------------------


def loss_curve(df, cfg, out_dir):
    """Two panels: the loss as it is, and the same on a log axis.

    `aleatoric_cov` is a log-likelihood, so its `0.5 log det Sigma` term
    drives the total negative once the predicted variances shrink -- which
    is what a converging run does. A log axis cannot show negative values
    at all, and symlog shows them but compresses the interesting decades
    into an unreadable band near zero.

    So: the left panel is plain linear, always correct and always
    complete. The right panel takes the sign of the final devel loss and
    plots the loss with that sign flipped in, which puts a converged run's
    values above zero and lets a real log axis resolve the late-time
    structure that the linear panel squashes.
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12.5, 4.5))

    # --- left: linear, everything, no transformation -------------------
    if df["train_loss"].notna().any():
        ax_lin.plot(df["epoch"], df["train_loss"], label="train", lw=1.6)
    ax_lin.plot(df["epoch"], df["devel_loss"], label="devel", lw=1.6)

    best_i = df["devel_loss"].idxmin()
    best_epoch = int(df["epoch"][best_i])
    best_loss = df["devel_loss"][best_i]
    ax_lin.axvline(best_epoch, color="0.4", ls="--", lw=1)
    ax_lin.plot(
        best_epoch,
        best_loss,
        "o",
        color="crimson",
        zorder=5,
        label=f"best devel {best_loss:.4f} @ {best_epoch}",
    )
    ax_lin.axhline(0, color="0.6", lw=0.8, zorder=0)
    ax_lin.set_xlabel("epoch")
    ax_lin.set_ylabel("aleatoric_cov loss")
    ax_lin.set_title("Loss (linear)")
    ax_lin.legend()
    ax_lin.grid(alpha=0.3)

    # --- right: log, sign taken from where the run ended ---------------
    devel = df["devel_loss"].dropna()
    sign = -1.0 if len(devel) and devel.iloc[-1] < 0 else 1.0
    tag = "-loss" if sign < 0 else "loss"

    dropped = 0
    for col, label in (("train_loss", "train"), ("devel_loss", "devel")):
        if col not in df or not df[col].notna().any():
            continue
        y = sign * df[col]
        keep = y > 0
        dropped += int((~keep & df[col].notna()).sum())
        ax_log.plot(df["epoch"][keep], y[keep], label=label, lw=1.6)

    if sign * best_loss > 0:
        ax_log.plot(
            best_epoch, sign * best_loss, "o", color="crimson", zorder=5
        )
        ax_log.axvline(best_epoch, color="0.4", ls="--", lw=1)

    ax_log.set_yscale("log")
    ax_log.set_xlabel("epoch")
    ax_log.set_ylabel(tag)
    title = f"Loss (log of {tag})"
    if dropped:
        title += f"  -- {dropped} points of opposite sign not shown"
    ax_log.set_title(title)
    ax_log.legend()
    ax_log.grid(alpha=0.3, which="both")

    return _save(fig, out_dir, "loss_curve.png")


def lr_schedule(df, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["epoch"], df["lr"], lw=1.6, drawstyle="steps-post")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate (backbone group)")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "lr_schedule.png")


def elementwise_mae(df, cfg, out_dir):
    names = _targets(cfg)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False
    )
    axes = axes[0]
    for ax, name in zip(axes, names):
        for prefix, linestyle in (("train", "-"), ("devel", "-")):
            col = f"{prefix}_mae_{name}"
            if col in df and df[col].notna().any():
                ax.plot(df["epoch"], df[col], linestyle, label=prefix, lw=1.5)
        ax.set_title(_pretty(name), fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE (scaled)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-target MAE", y=1.02)
    return _save(fig, out_dir, "elementwise_mae.png")


def epoch_time(df, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(df["epoch"], df["wall_seconds"], lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("wall seconds")
    ax.set_title("Epoch wall time (a rising trend means a starved dataloader)")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "epoch_time.png")


# --- model-based figures -----------------------------------------------------


def stn_crops(net, ds, cfg, out_dir, n=8, split="devel"):
    import torch

    n = min(n, len(ds.labels))
    batch = torch.stack([ds[i][0] for i in range(n)])
    with torch.no_grad():
        base = net.module if hasattr(net, "module") else net
        warped = base.spatial_transform(batch)

    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.8))
    for i in range(n):
        axes[0, i].imshow(batch[i, 0].numpy(), origin="lower", cmap="magma")
        axes[1, i].imshow(warped[i, 0].numpy(), origin="lower", cmap="magma")
        for row in (0, 1):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("after STN", fontsize=9)
    fig.suptitle("Spatial transformer input / output")
    return _save(fig, out_dir, f"{split}_stn_crops.png")


def stn_scales(net, ds, max_n=2000):
    """The STN's zoom parameter for the first ``max_n`` galaxies, in order.

    Split out from the histogram so the three STN figures share one pass
    over the localisation head instead of running it three times. Row order
    matches ``ds.data_info`` and ``_predict``, which is what lets the scale
    be plotted against a GALFIT column.
    """
    import torch

    base = net.module if hasattr(net, "module") else net
    n = min(len(ds.labels), max_n)
    vals = []
    with torch.no_grad():
        for start in range(0, n, 16):
            stop = min(start + 16, n)
            batch = torch.stack([ds[i][0] for i in range(start, stop)])
            xs = base.localization(batch).view(-1, base.fc_in_size)
            vals.append(base.fc_loc(xs).squeeze(-1).cpu().numpy())
    return np.concatenate(vals)


def stn_scale_hist(net, ds, cfg, out_dir, max_n=2000, s=None, split="devel"):
    """Distribution of the STN's single zoom parameter.

    A degenerate spike means the re-initialised ``fc_loc.0`` never learned
    anything and the transformer is passing its input straight through.
    """
    if s is None:
        s = stn_scales(net, ds, max_n=max_n)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(s, bins=50, color="steelblue", alpha=0.85)
    ax.axvline(1.0, color="crimson", ls="--", lw=1.2, label="identity (s = 1)")
    # Judge the spread relative to the zoom itself: an sd of 8e-4 sounds
    # small but is 0.3% of a mean of 0.28, i.e. the same crop for every
    # galaxy.  An absolute threshold misses that.
    spread = s.std() / abs(s.mean()) if s.mean() else np.inf
    degenerate = spread < 0.01

    ax.set_xlabel("STN zoom parameter s")
    ax.set_ylabel("count")
    ax.set_title(
        f"STN scale:  mean {s.mean():.4f}  sd {s.std():.4f}  "
        f"({100 * spread:.2f}% spread)"
        + (
            "\nNEAR-CONSTANT: the STN is not adapting per galaxy"
            if degenerate
            else ""
        )
    )
    ax.legend()
    return _save(fig, out_dir, f"{split}_stn_scale_hist.png")


def _info(ds, n):
    """The ``info.csv`` rows matching the first ``n`` dataset entries.

    ``_predict`` and ``stn_scales`` both walk the dataset in order from the
    start, so a plain head-slice of ``data_info`` lines up with what they
    return. Nothing here reorders, and it must stay that way.
    """
    return ds.data_info.iloc[:n]


def _physical(preds, truths, scaler):
    """Scaled network space -> the run's target space, for both arrays."""
    return scaler.inverse_transform(preds), scaler.inverse_transform(truths)


def pred_vs_true(preds, truths, scaler, cfg, out_dir, split="devel"):
    """G22 Fig. 11 / G23 Fig. 17: GALFIT value against predicted value."""
    names = _targets(cfg)
    p, t = _physical(preds, truths, scaler)

    fig, axes = style.panel_grid(len(names))
    for i, (ax, name) in enumerate(zip(axes, names)):
        tv, label, lim = display_space(name, t[:, i])
        pv, _, _ = display_space(name, p[:, i])

        style.hexbin(ax, tv, pv, extent=(lim + lim) if lim else None)
        lo = lim[0] if lim else float(min(tv.min(), pv.min()))
        hi = lim[1] if lim else float(max(tv.max(), pv.max()))
        style.identity_line(ax, lo, hi)
        if lim:
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
        ax.set_xlabel(f"GALFIT {label}")
        ax.set_ylabel(f"predicted {label}")
    fig.suptitle(f"{split.capitalize()}: predicted vs GALFIT", y=1.02)
    return _save(fig, out_dir, f"{split}_pred_vs_true.png")


def residual_hist(preds, truths, scaler, cfg, out_dir, split="devel"):
    """G22 Fig. 12 / G23 Fig. 18: residual histograms with boxed statistics."""
    names = _targets(cfg)
    p, t = _physical(preds, truths, scaler)

    fig, axes = style.panel_grid(len(names), height=3.6)
    for i, (ax, name) in enumerate(zip(axes, names)):
        r, label = style.residual(name, p[:, i], t[:, i])
        finite = r[np.isfinite(r)]
        ax.hist(finite, bins=45, color="slateblue", alpha=0.85)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        style.stats_box(ax, finite)
    fig.suptitle(f"{split.capitalize()} residuals", y=1.02)
    return _save(fig, out_dir, f"{split}_residual_hist.png")


def residual_vs_true(preds, truths, scaler, cfg, out_dir, split="devel"):
    """G22 Fig. 13: residuals against the true (GALFIT) value.

    The companion to the residual-vs-*predicted* figure in
    ``posterior_figures``. This one conditions on the truth, so it says
    where the model is wrong; that one conditions on the prediction, which
    is the only thing available at inference time and therefore the one
    that can define a quality flag.
    """
    names = _targets(cfg)
    p, t = _physical(preds, truths, scaler)

    fig, axes = style.panel_grid(len(names))
    for i, (ax, name) in enumerate(zip(axes, names)):
        tv, label, lim = display_space(name, t[:, i])
        r, rlabel = style.residual(name, p[:, i], t[:, i])
        ok = np.isfinite(tv) & np.isfinite(r)
        style.hexbin(ax, tv[ok], r[ok])
        style.zero_line(ax)
        style.overlay_percentiles(ax, tv[ok], r[ok])
        style.robust_ylim(ax, r[ok])
        ax.set_xlabel(f"GALFIT {label}")
        ax.set_ylabel(rlabel)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{split.capitalize()}: residual vs GALFIT value", y=1.02)
    return _save(fig, out_dir, f"{split}_residual_vs_true.png")


def fractional_residual_vs_observables(
    preds, truths, scaler, ds, cfg, out_dir, split="devel"
):
    """G22 Fig. 16: residuals against magnitude and against size.

    Both degrade for faint and for small galaxies, and this is the figure
    that says where. Magnitude comes from the catalog (``mag_sel``); size
    is the GALFIT ``R_e`` label rather than the prediction, because the
    point is to bin by a property the model did not choose.
    """
    names = _targets(cfg)
    p, t = _physical(preds, truths, scaler)
    info = _info(ds, len(p))

    axis_cols = [
        ("mag_sel", "catalog magnitude"),
        ("R_e_asec", 'GALFIT $R_e$ ["]'),
    ]
    axis_cols = [(c, lab) for c, lab in axis_cols if c in info.columns]
    if not axis_cols:
        raise ValueError("info.csv has neither mag_sel nor R_e_asec")

    fig, axes = plt.subplots(
        len(axis_cols),
        len(names),
        figsize=(4.3 * len(names), 3.8 * len(axis_cols)),
        squeeze=False,
    )
    for row, (col, xlabel) in enumerate(axis_cols):
        x = np.asarray(info[col], dtype=float)
        log_x = col == "R_e_asec"
        for i, name in enumerate(names):
            ax = axes[row, i]
            r, rlabel = style.residual(name, p[:, i], t[:, i])
            ok = np.isfinite(x) & np.isfinite(r) & (x > 0 if log_x else True)
            style.hexbin(
                ax, x[ok], r[ok], xscale="log" if log_x else None
            )
            style.zero_line(ax)
            style.overlay_percentiles(ax, x[ok], r[ok])
            style.robust_ylim(ax, r[ok])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(rlabel)
    fig.suptitle(f"{split.capitalize()}: residuals vs magnitude and size")
    return _save(fig, out_dir, f"{split}_residual_vs_observables.png")


def bt_residual_split(preds, truths, scaler, cfg, out_dir, split="devel"):
    """G22 Fig. 15, adapted: B/T residuals split by B/T regime.

    G22 split this by single- against double-component, which their
    *simulated* set supported by construction. Our labels are all
    two-component GALFIT fits and no ``bt`` in them is exactly 0 or 1
    (VIS z0 runs 0.0072 to 0.9953), so that split has no source here.

    The split that does mean something for us is the one G22 also drew and
    G23 acted on: the well-constrained middle band against the extremes,
    where GaMPEN's B/T is known to pile up and where the qualitative
    bulge-/disk-dominated labels replace the number.
    """
    names = _targets(cfg)
    if "custom_logit_bt" not in names:
        raise ValueError("this run has no B/T target")
    k = names.index("custom_logit_bt")
    p, t = _physical(preds, truths, scaler)

    from scipy.special import expit

    bt_true = expit(t[:, k])
    r, rlabel = style.residual("custom_logit_bt", p[:, k], t[:, k])

    mid = (bt_true > 0.1) & (bt_true < 0.85)
    subsets = [
        ("all", np.ones_like(mid, dtype=bool), "slateblue"),
        ("0.1 < B/T < 0.85", mid, "seagreen"),
        ("B/T < 0.1 or > 0.85", ~mid, "indianred"),
    ]

    fig, axes = style.panel_grid(len(subsets), width=4.0, height=3.6)
    for ax, (label, sel, colour) in zip(axes, subsets):
        vals = r[sel & np.isfinite(r)]
        if vals.size:
            ax.hist(vals, bins=40, color=colour, alpha=0.85)
            style.stats_box(ax, vals)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(rlabel)
        ax.set_ylabel("count")
    fig.suptitle(f"{split.capitalize()}: B/T residuals by regime", y=1.02)
    return _save(fig, out_dir, f"{split}_bt_residual_split.png")


def stn_extremes(net, ds, cfg, out_dir, n=6, s=None, max_n=2000, split="devel"):
    """G22 Fig. 5 / G23 Fig. 15: the most and least aggressive crops.

    A correctly-trained STN crops small galaxies hard and large ones
    barely, so the two rows should look visibly different in apparent
    size. If they do not, the head is not conditioning on the image.
    """
    import torch

    if s is None:
        s = stn_scales(net, ds, max_n=max_n)
    n = int(min(n, len(s) // 2))
    if n < 1:
        raise ValueError("not enough galaxies for an extremes montage")

    order = np.argsort(s)
    picks = [("lowest s\n(tightest crop)", order[:n]),
             ("highest s\n(widest crop)", order[-n:][::-1])]

    base = net.module if hasattr(net, "module") else net
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.3), squeeze=False)
    for row, (label, idxs) in enumerate(picks):
        batch = torch.stack([ds[int(i)][0] for i in idxs])
        with torch.no_grad():
            warped = base.spatial_transform(batch)
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(warped[col, 0].numpy(), origin="lower", cmap="magma")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"s = {s[idxs[col]]:.3f}", fontsize=7)
        axes[row, 0].set_ylabel(label, fontsize=8)
    # Say what s means in field terms. "Barely cropped" would be wrong here:
    # our trained s runs ~0.13-0.19, so even the widest crop keeps under a
    # fifth of the stamp. Each panel is auto-normalised independently, so
    # compare apparent *size*, not brightness, between the rows.
    fig.suptitle(
        "STN crops at the extremes of the zoom parameter\n"
        f"s = {s.min():.3f}-{s.max():.3f}, i.e. {s.min():.0%}-{s.max():.0%} "
        "of the stamp kept",
        fontsize=10,
    )
    return _save(fig, out_dir, f"{split}_stn_extremes.png")


def stn_scale_vs_re(net, ds, cfg, out_dir, s=None, max_n=2000, split="devel"):
    """The quantitative version of the STN extremes figure.

    Not in either paper, but it is the cleanest pass/fail on the one layer
    we re-initialise: ``s`` should rise with GALFIT ``R_e`` -- a big galaxy
    needs a wider field kept -- and a flat cloud means ``fc_loc.0`` learned
    nothing. Spearman is used rather than Pearson because only the
    monotonic trend is claimed, not its shape.
    """
    if s is None:
        s = stn_scales(net, ds, max_n=max_n)
    info = _info(ds, len(s))
    if "R_e_asec" not in info.columns:
        raise ValueError("info.csv has no R_e_asec column")
    re = np.asarray(info["R_e_asec"], dtype=float)

    ok = np.isfinite(re) & np.isfinite(s) & (re > 0)
    re, sv = re[ok], np.asarray(s)[ok]

    from scipy.stats import spearmanr

    rho, pval = spearmanr(re, sv)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(re, sv, s=6, alpha=0.35, color="steelblue", edgecolors="none")
    style.overlay_percentiles(ax, re, sv)
    ax.set_xscale("log")
    ax.set_xlabel('GALFIT $R_e$ ["]')
    ax.set_ylabel("STN zoom parameter s")

    # The trend is the point of this figure, and it lives in a narrow band of
    # s. Forcing the identity line into view flattens that band into a sliver
    # and hides the very thing being tested, so identity is drawn only when it
    # is already on scale and stated in words otherwise.
    pad = 0.15 * (sv.max() - sv.min() or 1.0)
    lo, hi = sv.min() - pad, sv.max() + pad
    if lo <= 1.0 <= hi:
        ax.axhline(1.0, color="crimson", ls="--", lw=1.0, label="identity (s = 1)")
    else:
        ax.set_ylim(lo, hi)
        ax.text(
            0.98,
            0.03,
            f"identity (s = 1) off scale;\nmedian crop keeps {np.median(sv):.0%} "
            "of the field",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

    ax.set_title(
        f"STN scale vs galaxy size\nSpearman $\\rho$ = {rho:+.3f}  "
        f"(p = {pval:.1e},  N = {sv.size})",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, out_dir, f"{split}_stn_scale_vs_re.png")


def input_pixel_hist(ds, cfg, out_dir, max_n=200, split="devel"):
    """arsinh-normalised Euclid pixels, against HSC if we have it.

    With ``pixel_zp = none`` the Euclid distribution is expected to sit about
    9.1x from HSC in VIS. That offset is fine and expected; what matters is
    whether the fine-tune converges. If training stalls or the loss curve is
    flat, re-export with ``--pixel-zp 27`` before touching anything else.
    """
    n = min(len(ds.labels), max_n)
    pix = np.concatenate([ds[i][0][0].numpy().ravel() for i in range(n)])

    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.hist(
        pix,
        bins=200,
        histtype="step",
        lw=1.6,
        density=True,
        label=f"Euclid {cfg['band']} (arsinh, pixel_zp={cfg.get('pixel_zp')})",
    )

    hsc = _find_hsc_demo()
    if hsc is not None:
        ax.hist(
            hsc,
            bins=200,
            histtype="step",
            lw=1.6,
            density=True,
            label="HSC tutorial cutouts (arsinh)",
        )
    else:
        ax.text(
            0.98,
            0.95,
            "HSC comparison set not on disk.\n"
            "Fetch with:  make hsc_demo  (in pipeline/GAMPEN)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9),
        )

    ax.set_yscale("log")
    ax.set_xlabel("arsinh-normalised pixel value")
    ax.set_ylabel("density")
    ax.set_title("Input dynamic range (gating diagnostic)")
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, out_dir, f"{split}_input_pixel_hist.png")


def _find_hsc_demo():
    """Return arsinh-normalised HSC demo pixels, or None if not downloaded."""
    from astropy.io import fits

    # `make hsc_demo` writes into the repo root by default; the data volume
    # is the other place it plausibly lives.
    here = Path(__file__).resolve()
    candidates = [
        layout.data_root() / "hsc" / "cutouts",
        here.parents[2] / "hsc" / "cutouts",
    ]
    for base in candidates:
        if not base.is_dir():
            continue
        files = sorted(base.glob("*.fits"))[:40]
        if not files:
            continue
        vals = []
        for f in files:
            with fits.open(f, memmap=False) as hdul:
                arr = np.asarray(hdul[0].data, dtype=float)
            vals.append(np.arcsinh(arr).ravel())
        return np.concatenate(vals)
    return None


def label_dists(cfg, scaler, out_dir):
    names = _targets(cfg)
    root = cfg.get("data_root")
    slug = splits.slug_for(cfg["seed"])
    frames = {}
    for split in ("train", "devel", "test"):
        try:
            frames[split] = cache_dataset.load_catalog(
                cfg["z_bin"], cfg["band"], split=split, slug=slug, root=root
            )
        except SystemExit:
            continue

    # squeeze=False keeps this 2-D even for a single-target run, where
    # matplotlib would otherwise return a 1-D array and the [row, col]
    # indexing below would go out of bounds.
    fig, axes = plt.subplots(
        2, len(names), figsize=(4.3 * len(names), 6.4), squeeze=False
    )
    for i, name in enumerate(names):
        for split, df in frames.items():
            raw = np.asarray(df[name], dtype=float)
            axes[0, i].hist(
                raw,
                bins=40,
                histtype="step",
                lw=1.4,
                density=True,
                label=split,
            )
            scaled = (raw - scaler.mean_[i]) / scaler.scale_[i]
            axes[1, i].hist(
                scaled,
                bins=40,
                histtype="step",
                lw=1.4,
                density=True,
                label=split,
            )
        axes[0, i].set_title(_pretty(name), fontsize=10)
        axes[0, i].set_xlabel("raw")
        axes[1, i].set_xlabel("scaled")
        for row in (0, 1):
            axes[row, i].legend(fontsize=8)
            axes[row, i].grid(alpha=0.3)
    fig.suptitle("Label distributions, raw and scaled", y=1.01)
    return _save(fig, out_dir, "label_dists.png")


# --- entry point -------------------------------------------------------------


def make_all(run_dir, out_dir=None, split="devel"):
    """Produce every figure that this run has the ingredients for.

    Figures land beside the run they describe, so a run directory is one
    self-contained thing. Resolution order is the explicit `out_dir`, then
    the run's recorded `figures_dir`, then `training_eval_figs/` inside the
    run directory.

    `split` defaults to devel on purpose. The test split must stay
    unlooked-at until a model is frozen -- every hyperparameter decision,
    dropout calibration included, is a validation activity -- so seeing it
    requires typing `--split test`.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())

    if out_dir is None:
        out_dir = cfg.get("figures_dir") or run_dir / "training_eval_figs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "metrics.csv")
    made, failed = [], []

    def attempt(name, fn, *a, **kw):
        try:
            made.append(fn(*a, **kw))
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log.warning("%s: skipped (%s)", name, exc)

    # metrics.csv only -- these describe the run, not a split.
    attempt("loss_curve", loss_curve, df, cfg, out_dir)
    attempt("lr_schedule", lr_schedule, df, cfg, out_dir)
    attempt("elementwise_mae", elementwise_mae, df, cfg, out_dir)
    attempt("epoch_time", epoch_time, df, cfg, out_dir)

    ds = scaler = net = None
    try:
        ds, scaler = rebuild_devel(cfg, split=split)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not rebuild the %s split (%s)", split, exc)

    if ds is not None:
        attempt("input_pixel_hist", input_pixel_hist, ds, cfg, out_dir,
                split=split)
        attempt("label_dists", label_dists, cfg, scaler, out_dir)
        try:
            net = load_best_model(cfg, run_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load best.pt (%s)", exc)

    if net is not None and ds is not None:
        # One pass over the localisation head feeds all three STN figures.
        scales = None
        try:
            scales = stn_scales(net, ds)
        except Exception as exc:  # noqa: BLE001
            log.warning("STN scales unavailable (%s)", exc)

        attempt("stn_crops", stn_crops, net, ds, cfg, out_dir, split=split)
        attempt("stn_scale_hist", stn_scale_hist, net, ds, cfg, out_dir,
                s=scales, split=split)
        attempt("stn_extremes", stn_extremes, net, ds, cfg, out_dir,
                s=scales, split=split)
        attempt("stn_scale_vs_re", stn_scale_vs_re, net, ds, cfg, out_dir,
                s=scales, split=split)

        try:
            preds, truths = _predict(net, ds)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s prediction failed (%s)", split, exc)
        else:
            common = (preds, truths, scaler, cfg, out_dir)
            attempt("pred_vs_true", pred_vs_true, *common, split=split)
            attempt("residual_hist", residual_hist, *common, split=split)
            attempt("residual_vs_true", residual_vs_true, *common, split=split)
            attempt("bt_residual_split", bt_residual_split, *common, split=split)
            attempt(
                "fractional_residual_vs_observables",
                fractional_residual_vs_observables,
                preds,
                truths,
                scaler,
                ds,
                cfg,
                out_dir,
                split=split,
            )

    log.info("figures: %d written to %s", len(made), out_dir)
    if failed:
        log.warning("figures skipped: %s", ", ".join(failed))
    return made


def main(argv=None):
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser(
        prog="python -m ggt.visualization.training_figures",
        description="Diagnostic figures for a stage-04 training run.",
    )
    ap.add_argument("run_dir")
    ap.add_argument(
        "--split",
        default="devel",
        help="split to draw (default devel; test only when a model is frozen)",
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    make_all(args.run_dir, out_dir=args.out_dir, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
