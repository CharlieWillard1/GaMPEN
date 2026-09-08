#!/usr/bin/env python
"""MC-dropout posterior figures for a finished run.

    python -m ggt.visualization.posterior_figures <run_dir> [--split devel]

Opt-in, and off by default in training, because it is the expensive tier:
`n_runs` stochastic forward passes over a whole split rather than one.
Everything the cheap tier in `training_figures` draws is a point estimate;
everything here needs the width of a posterior, which is the entire reason
GaMPEN exists.

**Why this does not call `ggt.modules.inference.predict`.** That function
rebuilds the architecture and reloads the checkpoint on every call, so five
hundred draws would mean five hundred model loads, and it constructs the
network through upstream's `model_factory` while our runs are built by
`ggt.models.transfer.build_model`. The MC loop lives here instead, with the
model built once -- and, importantly, with the draws taken batch-by-batch
while the images are already on the GPU, so the split is read from the cache
once rather than `n_runs` times. That is the difference between minutes and
hours.

Confidence intervals come from `result_aggregator.calculate_ci`, so an
interval means here exactly what it means in the published GaMPEN catalogs.
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
from ggt.visualization import style  # noqa: E402
from ggt.visualization import training_figures as tf  # noqa: E402

log = logging.getLogger(__name__)

_save = style.save

#: Physical prediction columns, in the order upstream's `.npy` PDF format
#: expects them. Kept identical to `unscale_preds` so our cached arrays are
#: directly comparable with the published G23 catalog's.
PHYSICAL_COLUMNS = (
    "preds_bt",
    "preds_R_e_asec",
    "preds_total_flux_adus",
    "preds_total_mag",
)

#: How each physical column is labelled and whether it reads better in log.
_AXIS = {
    "preds_bt": ("B/T", False),
    "preds_R_e_asec": ('$R_e$ ["]', True),
    "preds_total_flux_adus": ("total flux [ADU]", True),
    "preds_total_mag": ("total mag", False),
}

#: GALFIT truth column in `info.csv` for each prediction, where one exists.
_TRUTH_COLUMN = {
    "preds_bt": "bt",
    "preds_R_e_asec": "R_e_asec",
    "preds_total_flux_adus": "total_flux_adus",
    "preds_total_mag": "total_mag",
}

#: G23 §VI.4's qualitative B/T classes.
BT_DISK_MAX = 0.1
BT_BULGE_MIN = 0.85


# --- inference ---------------------------------------------------------------


def _pick_device():
    """The CUDA device with the most free memory, else CPU.

    These machines have two L40s shared with other jobs -- including, quite
    often, another of these. Always taking `cuda:0` means a second run dies
    on an out-of-memory error while the other card sits idle.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            free.append((torch.cuda.mem_get_info(i)[0], i))
        except Exception:  # noqa: BLE001
            free.append((0, i))
    free.sort(reverse=True)
    best_free, best = free[0]
    log.info("chose cuda:%d (%.1f GiB free)", best, best_free / 2**30)
    return f"cuda:{best}"


def _draw(preds, num_var):
    """One posterior draw per galaxy from a GaMPEN forward pass.

    **This is the step that makes the posterior a posterior.** GaMPEN's
    head emits more than a mean: with the `aleatoric_cov` loss its `n_out`
    is `3 * num_var` -- the means, the log-variances, and the strictly
    lower-triangular covariance terms -- and the prediction for a galaxy is
    a *draw* from the Gaussian those describe, not the mean of it.

    Taking only the means yields the epistemic (dropout) spread alone,
    which for our runs is roughly twenty times too narrow and produces
    coverage in the low single digits. The construction below is copied
    from `ggt.modules.inference`'s `--cov_errors` branch so the two agree
    exactly: `cov = L D L^T` with `L = I + N`.

    Falls back to the diagonal `--errors` form when `n_out` is
    `2 * num_var`, and to the bare mean when the model predicts no
    uncertainty at all.
    """
    import torch

    n_out = preds.shape[-1]

    if n_out == 3 * num_var:
        y_hat = preds[..., :num_var]
        var = torch.exp(preds[..., num_var : 2 * num_var])
        covs = preds[..., 2 * num_var :]

        D = torch.diag_embed(var)
        N = torch.zeros(preds.shape[0], num_var, num_var, device=preds.device)
        i, j = torch.tril_indices(num_var, num_var, -1)
        N[:, i, j] = covs

        Id = torch.eye(num_var, device=preds.device)
        Id = Id.reshape(1, num_var, num_var).repeat(preds.shape[0], 1, 1)
        L = Id + N
        cov_mat = torch.bmm(torch.bmm(L, D), torch.transpose(L, 1, 2))
        return torch.distributions.MultivariateNormal(y_hat, cov_mat).sample()

    if n_out == 2 * num_var:
        means = preds[..., :num_var]
        sigmas = torch.sqrt(torch.exp(preds[..., -num_var:]))
        return torch.distributions.Normal(means, sigmas).sample()

    return preds[..., :num_var]


def mc_dropout_samples(net, ds, cfg, n_runs=500, batch_size=64, device=None):
    """`(n_galaxies, n_runs, n_targets)` of scaled network outputs.

    `enable_dropout` is what makes this Bayesian: the model is in `eval()`
    so batch-norm behaves, but the dropout layers are switched back on by
    hand, and each pass is therefore a draw from a different thinned
    network. Without it every draw is identical and the posterior collapses
    to a point -- which the caller checks for.
    """
    import torch

    from ggt.utils.model_utils import enable_dropout, specify_dropout_rate

    if device is None:
        device = _pick_device()
    log.info("MC dropout: %d runs on %s", n_runs, device)

    rate = cfg.get("dropout")
    net = net.to(device)
    net.eval()
    if rate is not None:
        specify_dropout_rate(net, rate)
        log.info("dropout rate %g (from the run's config)", rate)
    else:
        log.warning("run config has no `dropout`; using the model's own rate")
    enable_dropout(net)

    n_gal = len(ds.labels)
    n_targets = ds.labels.shape[1]
    n_out = int(cfg.get("n_out") or n_targets)
    kind = {3: "aleatoric_cov (full covariance)", 2: "aleatoric (diagonal)"}.get(
        n_out // n_targets if n_targets else 0, "mean only (no aleatoric term)"
    )
    log.info("uncertainty model: %s  (n_out=%d, n_targets=%d)",
             kind, n_out, n_targets)
    out = np.empty((n_gal, n_runs, n_targets), dtype=np.float32)

    start = 0
    with torch.no_grad():
        while start < n_gal:
            stop = min(start + batch_size, n_gal)
            try:
                # Read the images once, then draw from them repeatedly
                # while they are already resident on the device.
                batch = torch.stack([ds[i][0] for i in range(start, stop)])
                batch = batch.to(device)
                for r in range(n_runs):
                    pred = _draw(net(batch), n_targets)
                    out[start:stop, r, :] = pred.cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                # Another job may have taken the card since we chose it, so
                # shrink and carry on rather than throwing away the passes
                # already done.
                torch.cuda.empty_cache()
                if batch_size <= 4:
                    raise
                batch_size = max(4, batch_size // 2)
                log.warning("CUDA OOM: retrying at batch size %d", batch_size)
                continue
            log.info("  %d/%d galaxies", stop, n_gal)
            start = stop
    return out


def to_physical(samples, scaler, cfg, train_bt=None, zp=None, pogson=None):
    """Scaled network space -> the four reportable physical quantities.

    Two inversions, in this order: the run's `StandardScaler`, then the
    logit/log label transforms. Doing them the other way round is silently
    wrong, and the round-trip test in `verify_round_trip` is what catches
    it.

    `train_bt` must be the *training* `bt` column: `expit_custom` uses it to
    re-derive the epsilons the forward transform used for exact 0s and 1s.
    In our Euclid labels no `bt` is exactly 0 or 1, so nothing is clipped --
    but passing it keeps us honest against a future dataset where some is.
    """
    from ggt.modules.result_aggregator import expit_custom
    from ggt.surveys.euclid import HSC_ZP, POGSON
    from ggt.utils.label_utils import flux_to_mag

    zp = HSC_ZP if zp is None else float(zp)
    pogson = POGSON if pogson is None else float(pogson)

    n_gal, n_runs, n_targets = samples.shape
    flat = scaler.inverse_transform(samples.reshape(-1, n_targets))
    names = list(cfg.get("target_columns") or [])

    out = {}
    if "custom_logit_bt" in names:
        col = flat[:, names.index("custom_logit_bt")]
        bt = expit_custom(col, scaling_array=train_bt)
        out["preds_bt"] = np.asarray(bt).reshape(n_gal, n_runs)
    if "ln_R_e_asec" in names:
        col = flat[:, names.index("ln_R_e_asec")]
        out["preds_R_e_asec"] = np.exp(col).reshape(n_gal, n_runs)
    if "ln_total_flux_adus" in names:
        col = flat[:, names.index("ln_total_flux_adus")]
        flux = np.exp(col).reshape(n_gal, n_runs)
        out["preds_total_flux_adus"] = flux
        out["preds_total_mag"] = flux_to_mag(flux, zp=zp, pogson=pogson)
    return out


def truths_physical(ds, cfg, n=None):
    """The GALFIT values for the same galaxies, in the same four spaces.

    Taken straight from `info.csv` rather than by inverting the scaled
    labels: these are the numbers GALFIT actually produced, and using them
    keeps the truth axis independent of any transform bug on the
    prediction side.
    """
    info = ds.data_info if n is None else ds.data_info.iloc[:n]
    out = {}
    for truth_col in set(_TRUTH_COLUMN.values()):
        if truth_col in info.columns:
            out[truth_col] = np.asarray(info[truth_col], dtype=float)
    return out


def verify_round_trip(ds, scaler, cfg, tol=1e-8):
    """Scale -> unscale on the labels themselves must return the originals.

    This isolates a transform bug from a model-quality problem: if the
    identity does not hold here, nothing downstream means what it says.
    """
    labels = np.asarray(ds.labels, dtype=float)
    fake = labels[:, None, :]  # one "draw" per galaxy
    got = to_physical(fake, scaler, cfg, train_bt=None)
    info = ds.data_info
    worst = {}
    for col, values in got.items():
        truth_col = _TRUTH_COLUMN.get(col)
        if truth_col is None or truth_col not in info.columns:
            continue
        want = np.asarray(info[truth_col], dtype=float)
        v = values[:, 0]
        ok = np.isfinite(v) & np.isfinite(want) & (np.abs(want) > 0)
        if not ok.any():
            continue
        rel = np.abs(v[ok] - want[ok]) / np.abs(want[ok])
        worst[col] = float(np.nanmax(rel))
    for col, err in sorted(worst.items()):
        verdict = "ok" if err < 1e-4 else "MISMATCH"
        log.info("round-trip %-24s max rel. error %.3e  %s", col, err, verdict)
    return worst


# --- caching -----------------------------------------------------------------


def cache_dir(run_dir, split):
    return Path(run_dir) / f"posteriors_{split}"


def _summarise(physical, object_ids, truths):
    """`preds_summary.csv` plus the per-galaxy KDE arrays.

    Column naming follows `result_aggregator.create_summary_df` exactly --
    `<col>_{mean,median,std,skew,kurtosis,mode,sig_ci,twosig_ci,threesig_ci}`
    -- so anything written against the published catalogs reads ours too.
    """
    import scipy.stats
    from ggt.modules.result_aggregator import calculate_ci

    df = pd.DataFrame({"object_id": np.asarray(object_ids)})
    for truth_col, values in truths.items():
        df[truth_col] = values

    x_domains, y_pdfs = {}, {}
    for col in PHYSICAL_COLUMNS:
        if col not in physical:
            continue
        arr = physical[col]  # (n_gal, n_runs)
        df[f"{col}_mean"] = arr.mean(axis=1)
        df[f"{col}_median"] = np.median(arr, axis=1)
        df[f"{col}_std"] = arr.std(axis=1)
        df[f"{col}_skew"] = scipy.stats.skew(arr, axis=1)
        df[f"{col}_kurtosis"] = scipy.stats.kurtosis(arr, axis=1)

        modes, sig, twosig, threesig = [], [], [], []
        xd, yp = [], []
        for i in range(arr.shape[0]):
            try:
                m, s1, s2, s3, x, y = calculate_ci(arr[i])
            except Exception:  # noqa: BLE001
                m = float(np.median(arr[i]))
                s1 = s2 = s3 = (np.nan, np.nan)
                x = y = np.full(1000, np.nan)
            modes.append(m)
            sig.append(s1)
            twosig.append(s2)
            threesig.append(s3)
            xd.append(x)
            yp.append(y)
        df[f"{col}_mode"] = modes
        df[f"{col}_sig_ci"] = sig
        df[f"{col}_twosig_ci"] = twosig
        df[f"{col}_threesig_ci"] = threesig
        # The 68.27% interval as a single width -- the "predicted sigma" the
        # residual diagnostics plot against.
        df[f"{col}_sig_width"] = [hi - lo for lo, hi in sig]
        x_domains[col] = np.asarray(xd)
        y_pdfs[col] = np.asarray(yp)
    return df, x_domains, y_pdfs


def write_cache(out_dir, df, x_domains, y_pdfs, samples):
    """Summary CSV, per-object PDF arrays in G23's format, and raw draws."""
    out_dir = Path(out_dir)
    (out_dir / "posterior_arrays").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "preds_summary.csv", index=False)

    cols = [c for c in PHYSICAL_COLUMNS if c in x_domains]
    for i, oid in enumerate(df["object_id"]):
        # Eight rows: four x-domains then four y-PDFs, matching
        # `result_aggregator.save_pdfs` so the files are interchangeable.
        stack = [x_domains[c][i] for c in cols] + [y_pdfs[c][i] for c in cols]
        np.save(out_dir / "posterior_arrays" / f"{int(oid)}.npy", np.stack(stack))

    # The raw draws, so coverage and the posterior panels can be recomputed
    # without paying for inference again.
    np.savez_compressed(
        out_dir / "samples.npz",
        object_id=np.asarray(df["object_id"]),
        columns=np.array(cols),
        **{c: samples[c].astype(np.float32) for c in cols},
    )
    log.info("cached posteriors to %s", out_dir)


def read_cache(out_dir):
    """`(summary_df, {column: (n_gal, n_runs)})`, or `None` if absent."""
    out_dir = Path(out_dir)
    csv, npz = out_dir / "preds_summary.csv", out_dir / "samples.npz"
    if not (csv.exists() and npz.exists()):
        return None
    df = pd.read_csv(csv)
    with np.load(npz, allow_pickle=False) as z:
        cols = [str(c) for c in z["columns"]]
        samples = {c: z[c] for c in cols}
    return df, samples


# --- figures -----------------------------------------------------------------


def _residual(col, pred, true):
    """Residual in the convention `style.residual` uses for that quantity.

    Sizes and fluxes are fractional; B/T and magnitude are differences.
    Magnitude is already logarithmic, so a difference in it *is* a ratio of
    fluxes -- making it fractional as well would double-count.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if col in ("preds_R_e_asec", "preds_total_flux_adus"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(true != 0, (pred - true) / true, np.nan)
    return pred - true


def _sigma(col, df):
    """The 68.27% interval width, in the same units as that column's residual.

    For sizes and fluxes the residual is fractional, so the width must be
    too -- an absolute width in ADU is dominated by how bright the galaxy
    is, and plotting it against a fractional residual mixes two scales.
    That mismatch alone flipped the flux correlation negative while
    magnitude, which is the same quantity in logarithmic units, came out
    strongly positive.

    Returns `(values, label)`.
    """
    label = _AXIS[col][0]
    width = np.asarray(df[f"{col}_sig_width"], dtype=float)
    if col in ("preds_R_e_asec", "preds_total_flux_adus"):
        mode = np.asarray(df[f"{col}_mode"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(mode != 0, width / mode, np.nan)
        return frac, f"fractional 68.27% width ({label})"
    return width, f"68.27% width ({label})"


def _resid_label(col):
    label, _ = _AXIS[col]
    if col in ("preds_R_e_asec", "preds_total_flux_adus"):
        return f"fractional {label} residual"
    return f"{label} residual"


def posterior_panels(df, samples, truths, out_dir, split="devel", n=4, seed=42):
    """G22 Fig. 10 / G23 Fig. 13: what one prediction actually looks like.

    Histogram of the draws, the KDE through them, 1/2/3-sigma shading in the
    papers' pink/yellow/green, the mode, and the GALFIT value. If the red
    GALFIT line sits outside the pink band more than about a third of the
    time, the posteriors are too narrow -- which is the calibration figure's
    job to quantify.
    """
    rng = np.random.default_rng(seed)
    cols = [c for c in PHYSICAL_COLUMNS if c in samples]
    picks = rng.choice(len(df), size=min(n, len(df)), replace=False)

    fig, axes = plt.subplots(
        len(picks), len(cols), figsize=(3.7 * len(cols), 2.7 * len(picks)),
        squeeze=False,
    )
    for row, gi in enumerate(picks):
        for k, col in enumerate(cols):
            ax = axes[row, k]
            draws = samples[col][gi]

            # Shading first and underneath: the intervals are nested, so each
            # is painted over the last, and all of them must sit behind the
            # histogram they describe. Drawn on top they hide it completely.
            for key, colour, label in (
                ("threesig_ci", "#c7e9c0", "3$\\sigma$"),
                ("twosig_ci", "#fee391", "2$\\sigma$"),
                ("sig_ci", "#fbb4b9", "1$\\sigma$"),
            ):
                lo, hi = _interval(df, col, key, gi)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.axvspan(lo, hi, color=colour, lw=0, zorder=0,
                               label=label if row == 0 and k == 0 else None)

            ax.hist(draws, bins=40, density=True, color="0.45",
                    edgecolor="none", alpha=0.65, zorder=2)

            from scipy.stats import gaussian_kde

            try:
                kde = gaussian_kde(draws)
                xs = np.linspace(draws.min(), draws.max(), 400)
                ax.plot(xs, kde(xs), color="k", lw=1.2, zorder=3)
            except Exception:  # noqa: BLE001
                pass

            mode = df[f"{col}_mode"].iloc[gi]
            ax.axvline(mode, color="k", ls="-", lw=1.4, zorder=4,
                       label="mode" if row == 0 and k == 0 else None)
            tcol = _TRUTH_COLUMN[col]
            if tcol in truths:
                ax.axvline(truths[tcol][gi], color="crimson", ls="--", lw=1.6,
                           zorder=4,
                           label="GALFIT" if row == 0 and k == 0 else None)
            ax.set_yticks([])
            ax.set_xlabel(_AXIS[col][0], fontsize=8)
            if k == 0:
                ax.set_ylabel(f"obj {int(df['object_id'].iloc[gi])}", fontsize=7)
    # A figure-level legend, so no panel loses data to it.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=len(labels),
               loc="upper center", bbox_to_anchor=(0.5, 0.975), frameon=False)
    fig.suptitle(f"{split.capitalize()}: MC-dropout posteriors", y=1.0)
    return _save(fig, out_dir, f"{split}_posterior_panels.png")


def _interval(df, col, key, i):
    """One `(lo, hi)` from a summary column, whether tuple or repr string."""
    v = df[f"{col}_{key}"].iloc[i]
    if isinstance(v, str):
        try:
            v = eval(v, {"__builtins__": {}}, {"nan": np.nan, "inf": np.inf})
        except Exception:  # noqa: BLE001
            return np.nan, np.nan
    try:
        return float(v[0]), float(v[1])
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


def corner_plot(df, samples, truths, out_dir, split="devel", index=None, seed=42):
    """G22 Fig. 9: the joint posterior for a single galaxy.

    The marginals on the diagonal are what every other figure summarises;
    the off-diagonals are what they throw away. Correlated errors between
    B/T and size are exactly the structure a per-parameter sigma cannot
    represent, which is why this figure is worth drawing once.
    """
    cols = [c for c in PHYSICAL_COLUMNS if c in samples and c != "preds_total_flux_adus"]
    explicit = index
    if index is None:
        # A galaxy with a truth for every parameter, chosen reproducibly.
        rng = np.random.default_rng(seed)
        index = int(rng.integers(len(df)))

    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(2.7 * k, 2.7 * k), squeeze=False)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            xi, yi = samples[cols[j]][index], samples[cols[i]][index]
            if i == j:
                ax.hist(xi, bins=40, color="steelblue", alpha=0.85, density=True)
                ax.set_yticks([])
                tcol = _TRUTH_COLUMN[cols[i]]
                if tcol in truths:
                    ax.axvline(truths[tcol][index], color="crimson", ls="--", lw=1.4)
                ax.axvline(df[f"{cols[i]}_mode"].iloc[index], color="k", lw=1.2)
            else:
                ax.hexbin(xi, yi, gridsize=25, cmap="Blues", mincnt=1)
                tx = _TRUTH_COLUMN[cols[j]]
                ty = _TRUTH_COLUMN[cols[i]]
                if tx in truths and ty in truths:
                    ax.plot(truths[tx][index], truths[ty][index], "*",
                            color="crimson", ms=13, mec="k", mew=0.5)
            if i == k - 1:
                ax.set_xlabel(_AXIS[cols[j]][0], fontsize=9)
            if j == 0 and i != 0:
                ax.set_ylabel(_AXIS[cols[i]][0], fontsize=9)
    oid = int(df["object_id"].iloc[index])
    fig.suptitle(f"Joint posterior, object {oid}  (red = GALFIT)", y=1.0)
    # A caller that named a galaxy wants that galaxy's file, not to
    # overwrite the run's default corner plot; several explicit indices in a
    # row would otherwise leave one figure.
    name = f"{split}_corner.png" if explicit is None else f"{split}_corner_{oid}.png"
    return _save(fig, out_dir, name)


def coverage_figure(samples, truths, out_dir, split="devel"):
    """G22 Fig. 8 / G23 Figs. 10, 16: claimed against achieved coverage.

    The diagonal is perfect calibration. Below it the posteriors are too
    narrow (overconfident), above it too wide. This is the figure that
    picks `dropout_rate`, and the mean deviation printed in the legend is
    the number to compare runs on.
    """
    by_target, truth_by_target = {}, {}
    for col in PHYSICAL_COLUMNS:
        tcol = _TRUTH_COLUMN.get(col)
        if col in samples and tcol in truths:
            by_target[col] = samples[col]
            truth_by_target[col] = truths[tcol]
    if not by_target:
        raise ValueError("no parameter has both draws and a GALFIT truth")

    pcts, curves, mean = cov.coverage_table(by_target, truth_by_target)

    fonts = style.COVERAGE_FONTS
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="perfect calibration")
    for col, achieved in curves.items():
        dev = cov.deviation_from_diagonal(pcts, achieved)
        ax.plot(pcts, achieved, marker="o", ms=3.5, lw=1.3,
                label=f"{_AXIS[col][0]}  ({dev:.1f}%)")
    dev = cov.deviation_from_diagonal(pcts, mean)
    ax.plot(pcts, mean, color="k", lw=2.2, label=f"mean  ({dev:.1f}%)")

    ax.set_xlabel("claimed confidence interval", fontsize=fonts["label"])
    ax.set_ylabel("fraction of galaxies containing the GALFIT value",
                  fontsize=fonts["label"])
    ax.tick_params(labelsize=fonts["tick"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(
        f"{split.capitalize()} coverage  (legend: mean deviation)",
        fontsize=fonts["title"],
    )
    ax.legend(fontsize=fonts["legend"], loc="lower right")
    return _save(fig, out_dir, f"{split}_coverage.png")


def residual_vs_predicted(df, truths, out_dir, split="devel"):
    """G22 Fig. 14 / G23 Figs. 19, 30, 31 -- the quality-flag figure.

    The counterpart in `training_figures` conditions on the truth, which
    tells you where the model is wrong but is useless at inference time.
    This one conditions on the *prediction*, the only thing available for a
    galaxy with no GALFIT fit, and so it is what a quality flag can
    actually be written against.
    """
    cols = [c for c in PHYSICAL_COLUMNS if f"{c}_mode" in df.columns]
    fig, axes = style.panel_grid(len(cols))
    for ax, col in zip(axes, cols):
        pred = np.asarray(df[f"{col}_mode"], dtype=float)
        tcol = _TRUTH_COLUMN[col]
        if tcol not in truths:
            raise ValueError(f"no GALFIT truth column `{tcol}` for {col}")
        r = _residual(col, pred, truths[tcol])
        label, log_x = _AXIS[col]
        ok = np.isfinite(pred) & np.isfinite(r) & ((pred > 0) if log_x else True)
        style.hexbin(ax, pred[ok], r[ok], xscale="log" if log_x else None)
        style.zero_line(ax)
        style.overlay_percentiles(ax, pred[ok], r[ok])
        style.robust_ylim(ax, r[ok])
        ax.set_xlabel(f"predicted {label}")
        ax.set_ylabel(_resid_label(col))
    fig.suptitle(f"{split.capitalize()}: residual vs PREDICTED value")
    return _save(fig, out_dir, f"{split}_residual_vs_predicted.png")


def sigma_vs_predicted(df, out_dir, split="devel"):
    """G22 Fig. 17 / G23 Figs. 20, 32, 33: predicted sigma vs prediction.

    Sigma is the width of the 68.27% interval. Where it widens is where the
    model is telling you not to trust it, and that shape -- not the mean
    accuracy -- is what makes a posterior worth having.
    """
    cols = [c for c in PHYSICAL_COLUMNS if f"{c}_sig_width" in df.columns]
    fig, axes = style.panel_grid(len(cols))
    for ax, col in zip(axes, cols):
        pred = np.asarray(df[f"{col}_mode"], dtype=float)
        sig, siglabel = _sigma(col, df)
        label, log_x = _AXIS[col]
        ok = np.isfinite(pred) & np.isfinite(sig) & ((pred > 0) if log_x else True)
        style.hexbin(ax, pred[ok], sig[ok], xscale="log" if log_x else None)
        style.overlay_percentiles(ax, pred[ok], sig[ok])
        style.robust_ylim(ax, sig[ok])
        ax.set_xlabel(f"predicted {label}")
        ax.set_ylabel(siglabel)
    fig.suptitle(f"{split.capitalize()}: predicted $\\sigma$ vs prediction")
    return _save(fig, out_dir, f"{split}_sigma_vs_predicted.png")


def sigma_vs_residual(df, truths, out_dir, split="devel"):
    """G22 Fig. 18: does the claimed uncertainty grow where it is wrong?

    The check that sigma is not decorative. A rising trend means the model
    knows which galaxies it is guessing at; a flat one means the width is
    uninformative even if the coverage happens to come out right.
    """
    cols = [c for c in PHYSICAL_COLUMNS if f"{c}_sig_width" in df.columns]
    fig, axes = style.panel_grid(len(cols))
    for ax, col in zip(axes, cols):
        tcol = _TRUTH_COLUMN[col]
        if tcol not in truths:
            raise ValueError(f"no GALFIT truth column `{tcol}` for {col}")
        pred = np.asarray(df[f"{col}_mode"], dtype=float)
        sig, siglabel = _sigma(col, df)
        r = np.abs(_residual(col, pred, truths[tcol]))
        ok = np.isfinite(sig) & np.isfinite(r)
        style.hexbin(ax, sig[ok], r[ok])
        style.overlay_percentiles(ax, sig[ok], r[ok])
        style.robust_ylim(ax, r[ok])
        from scipy.stats import spearmanr

        rho, _ = spearmanr(sig[ok], r[ok])
        ax.set_xlabel(siglabel)
        ax.set_ylabel(f"|{_resid_label(col)}|")
        ax.set_title(f"Spearman $\\rho$ = {rho:+.3f}", fontsize=9)
    fig.suptitle(f"{split.capitalize()}: predicted $\\sigma$ vs |residual|")
    return _save(fig, out_dir, f"{split}_sigma_vs_residual.png")


def bt_qualitative(df, truths, out_dir, split="devel"):
    """G22 Figs. 19 and 20: the quantitative -> qualitative transformation.

    GaMPEN's B/T piles up at the extremes, where the number it reports is
    less meaningful than the statement "bulge-dominated" or
    "disk-dominated". The left panel shows what replacing extreme
    predictions with the class label does to the residuals; the right is
    the confusion matrix for those classes.
    """
    if "preds_bt_mode" not in df.columns or "bt" not in truths:
        raise ValueError("no B/T predictions or truths")
    pred = np.asarray(df["preds_bt_mode"], dtype=float)
    true = np.asarray(truths["bt"], dtype=float)

    extreme = (pred < BT_DISK_MAX) | (pred > BT_BULGE_MIN)
    snapped = np.where(pred < BT_DISK_MAX, 0.0,
                       np.where(pred > BT_BULGE_MIN, 1.0, pred))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    ax.hist(pred[extreme] - true[extreme], bins=40, alpha=0.7,
            color="slateblue", label="as predicted")
    ax.hist(snapped[extreme] - true[extreme], bins=40, alpha=0.7,
            color="darkorange", label="snapped to 0 / 1")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("B/T residual")
    ax.set_ylabel("count")
    ax.set_title(
        f"Extreme predictions only (N = {int(extreme.sum())})", fontsize=10
    )
    ax.legend(fontsize=8)

    labels = ["disk\n(B/T < 0.1)", "intermediate", "bulge\n(B/T > 0.85)"]

    def _cls(v):
        return np.where(v < BT_DISK_MAX, 0, np.where(v > BT_BULGE_MIN, 2, 1))

    ct, pt = _cls(true), _cls(pred)
    cm = np.zeros((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            cm[a, b] = np.sum((ct == a) & (pt == b))
    rows = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)

    ax = axes[1]
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{frac[a, b]:.2f}\n({int(cm[a, b])})",
                    ha="center", va="center", fontsize=8,
                    color="white" if frac[a, b] > 0.55 else "black")
    ax.set_xticks(range(3), labels, fontsize=7)
    ax.set_yticks(range(3), labels, fontsize=7)
    ax.set_xlabel("GaMPEN class")
    ax.set_ylabel("GALFIT class")
    # Raw accuracy on these three classes is dominated by the intermediate
    # bin, which holds most of the sample. A model that answers
    # "intermediate" every time scores the majority base rate, so quote that
    # alongside -- otherwise 0.91 reads as good when it means nothing was
    # learned about the extremes.
    total = cm.sum()
    acc = np.trace(cm) / total if total else np.nan
    baseline = cm.sum(axis=0).max() / total if total else np.nan
    per_class = np.divide(np.diag(cm), cm.sum(axis=1),
                          out=np.full(3, np.nan), where=cm.sum(axis=1) > 0)
    balanced = np.nanmean(per_class)
    ax.set_title(
        f"Row-normalised confusion\n"
        f"accuracy {acc:.2f}  (majority baseline {baseline:.2f}, "
        f"balanced {balanced:.2f})",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{split.capitalize()}: qualitative B/T classes")
    return _save(fig, out_dir, f"{split}_bt_qualitative.png")


# --- driver ------------------------------------------------------------------


def compute(run_dir, split="devel", n_runs=500, batch_size=64, force=False,
            limit=None, device=None):
    """Run (or reuse) MC-dropout inference and return `(df, samples, truths)`.

    The cache is keyed on the run and the split. Re-running is free unless
    `force` is passed, which is what makes the figures cheap to iterate on
    once the expensive part has been paid for.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    out_dir = cache_dir(run_dir, split)

    ds, scaler = tf.rebuild_devel(cfg, split=split)
    if limit:
        n = min(limit, len(ds.labels))
        ds.labels = ds.labels[:n]
        ds.observations = ds.observations[:n]
        ds.filenames = ds.filenames[:n]
        ds.data_info = ds.data_info.iloc[:n]

    truths = truths_physical(ds, cfg)

    cached = None if force else read_cache(out_dir)
    if cached is not None and len(cached[0]) == len(ds.labels):
        log.info("reusing cached posteriors in %s", out_dir)
        df, samples = cached
        return df, samples, truths
    if cached is not None:
        log.warning("cache has %d rows but the split has %d; recomputing",
                    len(cached[0]), len(ds.labels))

    verify_round_trip(ds, scaler, cfg)

    net = tf.load_best_model(cfg, run_dir)
    raw = mc_dropout_samples(net, ds, cfg, n_runs=n_runs,
                             batch_size=batch_size, device=device)

    # The guard that matters: if every draw is identical, `enable_dropout`
    # did not take and every "posterior" is a point mass wearing error bars.
    spread = raw.std(axis=1)
    if not np.all(np.isfinite(spread)) or spread.max() <= 0:
        raise RuntimeError(
            "MC-dropout draws have zero spread -- dropout was not active, so "
            "every run collapsed to the same forward pass"
        )
    log.info("draw spread (scaled): min %.3e  median %.3e  max %.3e",
             spread.min(), np.median(spread), spread.max())

    train_bt = _training_bt(cfg)
    physical = to_physical(raw, scaler, cfg, train_bt=train_bt)
    df, x_domains, y_pdfs = _summarise(
        physical, np.asarray(ds.data_info["object_id"]), truths
    )
    write_cache(out_dir, df, x_domains, y_pdfs, physical)
    return df, physical, truths


def _training_bt(cfg):
    """The training split's `bt` column, for `expit_custom`'s epsilons.

    Read straight off the split CSV. Building the training dataset to get
    at one column would mean loading 27,000 cutouts from the cache to
    recover a number already sitting in a text file.
    """
    try:
        from ggt.data import layout, splits

        path = (
            layout.splits_dir(cfg["z_bin"], cfg["band"], cfg.get("data_root"))
            / f"{splits.slug_for(cfg['seed'])}-train.csv"
        )
        return np.asarray(pd.read_csv(path)["bt"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the training bt column (%s); "
                    "expit_custom will not clip at the endpoints", exc)
        return None


def make_all(run_dir, out_dir=None, split="devel", n_runs=500, batch_size=64,
             force=False, limit=None, device=None):
    """Every posterior figure this run has the ingredients for."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    if out_dir is None:
        out_dir = cfg.get("figures_dir") or run_dir / "training_eval_figs"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, samples, truths = compute(
        run_dir, split=split, n_runs=n_runs, batch_size=batch_size,
        force=force, limit=limit, device=device,
    )

    made, failed = [], []

    def attempt(name, fn, *a, **kw):
        try:
            made.append(fn(*a, **kw))
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log.warning("%s: skipped (%s)", name, exc)

    attempt("posterior_panels", posterior_panels, df, samples, truths, out_dir,
            split=split)
    attempt("corner_plot", corner_plot, df, samples, truths, out_dir, split=split)
    attempt("coverage", coverage_figure, samples, truths, out_dir, split=split)
    attempt("residual_vs_predicted", residual_vs_predicted, df, truths, out_dir,
            split=split)
    attempt("sigma_vs_predicted", sigma_vs_predicted, df, out_dir, split=split)
    attempt("sigma_vs_residual", sigma_vs_residual, df, truths, out_dir,
            split=split)
    attempt("bt_qualitative", bt_qualitative, df, truths, out_dir, split=split)

    log.info("posterior figures: %d written to %s", len(made), out_dir)
    if failed:
        log.warning("posterior figures skipped: %s", ", ".join(failed))
    return made


def main(argv=None):
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    ap = argparse.ArgumentParser(
        prog="python -m ggt.visualization.posterior_figures",
        description="MC-dropout posterior figures for a stage-04 run.",
    )
    ap.add_argument("run_dir")
    ap.add_argument(
        "--split",
        default="devel",
        help="split to draw (default devel; test only when a model is frozen)",
    )
    ap.add_argument("--n-runs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N galaxies, for a quick check")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a cache is present")
    ap.add_argument("--device", default=None,
                    help="e.g. cuda:1 or cpu; default is the emptiest GPU")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    make_all(args.run_dir, out_dir=args.out_dir, split=args.split,
             n_runs=args.n_runs, batch_size=args.batch_size,
             force=args.force, limit=args.limit, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
