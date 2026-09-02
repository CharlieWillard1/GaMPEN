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

log = logging.getLogger(__name__)

DPI = 130


# --- shared helpers ----------------------------------------------------------


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def _targets(cfg):
    return list(cfg.get("target_columns") or [])


def _pretty(name):
    return name.replace("_", " ")


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
        for prefix, style in (("train", "-"), ("devel", "-")):
            col = f"{prefix}_mae_{name}"
            if col in df and df[col].notna().any():
                ax.plot(df["epoch"], df[col], style, label=prefix, lw=1.5)
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


def stn_crops(net, ds, cfg, out_dir, n=8):
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
    return _save(fig, out_dir, "stn_crops.png")


def stn_scale_hist(net, ds, cfg, out_dir, max_n=2000):
    """Distribution of the STN's single zoom parameter.

    A degenerate spike means the re-initialised ``fc_loc.0`` never learned
    anything and the transformer is passing its input straight through.
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
    s = np.concatenate(vals)

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
    return _save(fig, out_dir, "stn_scale_hist.png")


def display_space(name, values):
    """Map a target into the units it is easiest to judge in.

    The network trains on `custom_logit_bt` because a logit is unbounded
    and symmetric, which suits a Gaussian likelihood -- but bulge-to-total
    is a fraction, and a plot of it should run 0 to 1.

    `expit` is the plain inverse; `logit_custom` additionally nudges exact
    0s and 1s inward by an epsilon derived from the training column, so
    values that started at exactly 0 or 1 come back very close to, but not
    exactly at, the endpoints. That is immaterial for a diagnostic plot.

    Returns `(values, label, limits)`; limits is None to autoscale.
    """
    if name == "custom_logit_bt":
        from scipy.special import expit

        return expit(values), "B/T", (0.0, 1.0)
    return values, _pretty(name), None


def devel_pred_vs_true(preds, truths, scaler, cfg, out_dir):
    names = _targets(cfg)
    p = scaler.inverse_transform(preds)
    t = scaler.inverse_transform(truths)

    fig, axes = plt.subplots(
        1, len(names), figsize=(4.3 * len(names), 4.0), squeeze=False
    )
    axes = axes[0]
    for i, (ax, name) in enumerate(zip(axes, names)):
        tv, label, lim = display_space(name, t[:, i])
        pv, _, _ = display_space(name, p[:, i])

        ax.hexbin(
            tv,
            pv,
            gridsize=35,
            cmap="viridis",
            mincnt=1,
            extent=(lim + lim) if lim else None,
        )
        lo = lim[0] if lim else min(tv.min(), pv.min())
        hi = lim[1] if lim else max(tv.max(), pv.max())
        ax.plot([lo, hi], [lo, hi], "w--", lw=1.2)
        if lim:
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
        ax.set_xlabel(f"GALFIT {label}")
        ax.set_ylabel(f"predicted {label}")
    fig.suptitle("Devel: predicted vs GALFIT", y=1.02)
    return _save(fig, out_dir, "devel_pred_vs_true.png")


def devel_residual_hist(preds, truths, scaler, cfg, out_dir):
    names = _targets(cfg)
    resid = scaler.inverse_transform(preds) - scaler.inverse_transform(truths)

    fig, axes = plt.subplots(
        1, len(names), figsize=(4.3 * len(names), 3.6), squeeze=False
    )
    axes = axes[0]
    for i, (ax, name) in enumerate(zip(axes, names)):
        r = resid[:, i]
        ax.hist(r, bins=45, color="slateblue", alpha=0.85)
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel(f"pred - GALFIT  ({_pretty(name)})")
        ax.set_ylabel("count")
        ax.text(
            0.03,
            0.97,
            f"$\\mu$ {r.mean():+.4f}\nmed {np.median(r):+.4f}\n"
            f"$\\sigma$ {r.std():.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
    fig.suptitle("Devel residuals", y=1.02)
    return _save(fig, out_dir, "devel_residual_hist.png")


def input_pixel_hist(ds, cfg, out_dir, max_n=200):
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
    return _save(fig, out_dir, "input_pixel_hist.png")


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


def make_all(run_dir, out_dir=None):
    """Produce every figure that this run has the ingredients for.

    Figures land beside the run they describe, so a run directory is one
    self-contained thing. Resolution order is the explicit `out_dir`, then
    the run's recorded `figures_dir`, then `training_eval_figs/` inside the
    run directory.
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

    attempt("loss_curve", loss_curve, df, cfg, out_dir)
    attempt("lr_schedule", lr_schedule, df, cfg, out_dir)
    attempt("elementwise_mae", elementwise_mae, df, cfg, out_dir)
    attempt("epoch_time", epoch_time, df, cfg, out_dir)

    ds = scaler = net = None
    try:
        ds, scaler = rebuild_devel(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not rebuild the devel split (%s)", exc)

    if ds is not None:
        attempt("input_pixel_hist", input_pixel_hist, ds, cfg, out_dir)
        attempt("label_dists", label_dists, cfg, scaler, out_dir)
        try:
            net = load_best_model(cfg, run_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load best.pt (%s)", exc)

    if net is not None and ds is not None:
        attempt("stn_crops", stn_crops, net, ds, cfg, out_dir)
        attempt("stn_scale_hist", stn_scale_hist, net, ds, cfg, out_dir)
        try:
            preds, truths = _predict(net, ds)
            attempt(
                "devel_pred_vs_true",
                devel_pred_vs_true,
                preds,
                truths,
                scaler,
                cfg,
                out_dir,
            )
            attempt(
                "devel_residual_hist",
                devel_residual_hist,
                preds,
                truths,
                scaler,
                cfg,
                out_dir,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("devel prediction failed (%s)", exc)

    log.info("figures: %d written to %s", len(made), out_dir)
    if failed:
        log.warning("figures skipped: %s", ", ".join(failed))
    return made


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(
            "usage: python -m ggt.visualization.training_figures <run_dir>"
        )
    make_all(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
