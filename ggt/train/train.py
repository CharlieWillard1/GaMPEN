#!/usr/bin/env python
"""Fine-tune a GaMPEN model on one (z_bin, band) Euclid dataset.

    python -m ggt.train.train --z-bin 0 --band VIS --run-name my_run

This replaces the original GaMPEN training entry point, which hardcoded a
strict ``load_state_dict`` (incompatible with transferring weights across an
input-size change), a single SGD parameter group, mandatory MLflow, and
``--parallel`` defaulting to on. The order of operations still follows it
closely, and every generic building block -- the network, the loss, the
metrics, the dataset -- is unchanged.

What it adds: the pixel-zeropoint guard, create-once/reuse splits and a
persisted scaler, checkpoint surgery with a re-initialisation whitelist,
per-group learning rates, and a `config.json` a run can be reconstructed
from.

Each step is a separate importable function so that analysis notebooks can
rebuild the identical datasets and model without training anything.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np

from ggt.data import cache_dataset, layout, splits
from ggt.models import transfer as models
from ggt.surveys import euclid

log = logging.getLogger(__name__)

# The pretrained head's output order.  Never permute.
TARGET_COLUMNS = list(euclid.TARGET_COLUMNS)


# --- provenance --------------------------------------------------------------


def git_describe(repo_dir):
    """``(sha, branch)`` for a git checkout, or ``(None, None)``."""

    def run(*args):
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    return run("rev-parse", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def enclosing_repo(start):
    """Nearest ancestor containing `.git`, or None.

    A git submodule's `.git` is a *file*, not a directory, so test for
    existence rather than `is_dir()`. Walking up beats hardcoded
    `parents[n]` arithmetic, which silently returns the wrong directory the
    moment a module moves.
    """
    p = Path(start).resolve()
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


# --- 1. paths ----------------------------------------------------------------


def resolve(z_bin, band, run_name, root=None, figures_dir=None):
    """Resolve every path this run touches, failing early if data is absent.

    Everything a run produces -- weights, metrics, logs and figures --
    lands together under the data volume's runs root, so a run is one
    self-contained directory. See `ggt.data.layout`.
    """
    if band not in euclid.BANDS:
        raise SystemExit(
            f"unknown band {band!r}; expected one of {euclid.BANDS}"
        )

    info = layout.info_csv(z_bin, band, root)
    if not info.exists():
        raise SystemExit(
            f"no info.csv at {info}\n"
            f"Build the pixel cache and this band's labels first:\n"
            f"  python -m euclid_prep.build_cache --z-bin {z_bin} "
            f"--subset-catalog <subset.fits>\n"
            f"  python -m euclid_prep.build_info  --z-bin {z_bin} "
            f"--band {band} --subset-catalog <subset.fits> "
            f"--galfit-catalog <galfit_results.fits>"
        )

    run = layout.ensure(layout.run_dir(z_bin, band, run_name, root))
    figs = (
        Path(figures_dir)
        if figures_dir
        else run / "training_eval_figs"
    )
    return {
        "info_csv": info,
        "band_dir": layout.band_dir(z_bin, band, root),
        "cache_manifest": layout.cache_manifest_path(z_bin, root),
        "run_dir": run,
        "figures_dir": figs,
        "metrics_csv": run / "metrics.csv",
        "train_log": run / "train.log",
        "config_json": run / "config.json",
    }


# --- 2. the pixel-zeropoint guard --------------------------------------------


def check_pixel_zp(z_bin, pixel_zp, root=None):
    """Refuse to train if the cache was not built with the requested ZP.

    There is no FITS export any more, so the convention lives in
    ``cache_manifest.json``. Training on differently-scaled pixels than you
    believe -- a factor of 9.1 in VIS, ~6 mag in the NIR bands -- produces a
    model that trains perfectly well and means nothing.
    """
    manifest_path = layout.cache_manifest_path(z_bin, root)
    if not manifest_path.exists():
        raise SystemExit(f"no cache manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    built = manifest.get("pixel_zp")

    if (built is None) != (pixel_zp is None) or (
        built is not None
        and pixel_zp is not None
        and not np.isclose(float(built), float(pixel_zp))
    ):
        raise SystemExit(
            "PIXEL_ZP mismatch -- refusing to start.\n"
            f"  requested --pixel-zp : {pixel_zp!r}\n"
            f"  cache was built with : {built!r}\n"
            f"  cache manifest       : {manifest_path}\n"
            "Rebuild the cache with a matching --pixel-zp, or pass the value "
            "the cache actually holds."
        )
    log.info("pixel_zp %r agrees with the cache manifest", built)
    return manifest


# --- 3. splits and scaler ----------------------------------------------------


def prepare_split(
    z_bin,
    band,
    seed=42,
    fractions=None,
    balance_on="bt",
    force_resplit=False,
    root=None,
):
    """Create the split and scaler once; reuse and verify them thereafter."""
    manifest = splits.build(
        z_bin,
        band,
        seed=seed,
        fractions=fractions,
        balance_on=balance_on,
        root=root,
        force_resplit=force_resplit,
        target_metrics=TARGET_COLUMNS,
    )
    sizes = manifest["sizes"]
    log.info(
        "split %s (train/devel/test = %d/%d/%d, hashes %s)",
        manifest["slug"],
        sizes["train"],
        sizes["devel"],
        sizes["test"],
        "/".join(
            manifest["hashes"][s][:8] for s in ("train", "devel", "test")
        ),
    )

    scaler = splits.load_scaler(
        z_bin, band, seed=seed, root=root, target_metrics=TARGET_COLUMNS
    )
    for name, mean, scale in zip(TARGET_COLUMNS, scaler.mean_, scaler.scale_):
        log.info("scaler %-24s mean %+.6f  scale %.6f", name, mean, scale)
    return manifest, scaler


# --- 4. datasets and loaders -------------------------------------------------


def truncate(ds, n):
    """Keep only the first `n` rows (smoke / overfit tests)."""
    n = min(n, len(ds.labels))
    ds.labels = ds.labels[:n]
    ds.observations = ds.observations[:n]
    ds.filenames = ds.filenames[:n]
    ds.data_info = ds.data_info.iloc[:n]
    return ds


def build_loaders(
    z_bin,
    band,
    scaler,
    seed=42,
    cutout_size=None,
    channels=3,
    repeat_dims=True,
    normalize=True,
    transform=True,
    expand_data=1,
    batch_size=16,
    n_workers=8,
    root=None,
    limit=None,
):
    """Three datasets sharing one scaler, plus a clean train-eval loader.

    The train split gets the augmentation stack; devel and test get a bare
    centre-crop. A fourth loader re-reads the *train* split without
    augmentation or expansion, because a train loss measured through random
    rotations cannot be compared against the devel loss.
    """
    import kornia.augmentation as K
    import torch.nn as nn
    from ggt.data import get_data_loader

    if cutout_size is None:
        cutout_size = euclid.target_crop_px(z_bin)
    slug = splits.slug_for(seed)

    crop_only = nn.Sequential(K.CenterCrop(cutout_size))
    augmented = nn.Sequential(
        K.CenterCrop(cutout_size),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        K.RandomRotation(360),
    )

    def make(split, tfm, expand):
        ds = cache_dataset.make_dataset(
            z_bin,
            band,
            split=split,
            slug=slug,
            root=root,
            target_metrics=TARGET_COLUMNS,
            cutout_size=cutout_size,
            channels=channels,
            repeat_dims=repeat_dims,
            normalize=normalize,
            transform=tfm,
            expand_factor=expand,
            scaler=scaler,
        )
        return truncate(ds, limit) if limit else ds

    train_tfm = augmented if transform else crop_only
    datasets = {
        "train": make("train", train_tfm, expand_data),
        "devel": make("devel", crop_only, 1),
        "test": make("test", crop_only, 1),
    }

    # The train-eval view differs from the train set only in its transform and
    # expansion, so share the loaded pixels rather than reading them a second
    # time -- that is another ~7.5 GB of resident memory on a full 12k bin.
    train_eval = copy.copy(datasets["train"])
    train_eval.transform = crop_only
    train_eval.expand_factor = 1

    loaders = {
        "train": get_data_loader(
            datasets["train"], batch_size, n_workers, shuffle=True
        ),
        "devel": get_data_loader(
            datasets["devel"], batch_size, n_workers, shuffle=False
        ),
        "test": get_data_loader(
            datasets["test"], batch_size, n_workers, shuffle=False
        ),
    }
    train_eval_loader = get_data_loader(
        train_eval, batch_size, n_workers, shuffle=False
    )

    sizes = {k: len(v) for k, v in datasets.items()}
    log.info(
        "datasets: train %d (x%d augmented) / devel %d / test %d, "
        "cutout_size %d",
        sizes["train"],
        expand_data,
        sizes["devel"],
        sizes["test"],
        cutout_size,
    )
    return loaders, train_eval_loader, sizes, cutout_size


# --- 5. model ----------------------------------------------------------------


def build_net(
    z_bin,
    cutout_size,
    init_from="real",
    dropout=None,
    channels=3,
    freeze="none",
    allow_broad_reinit=False,
    root=None,
    parallel=None,
):
    """Build, load the HSC checkpoint into, and freeze the network."""
    import torch
    import torch.nn as nn

    n_out = int((3 * len(TARGET_COLUMNS) + len(TARGET_COLUMNS) ** 2) / 2)
    if dropout is None:
        dropout = models.default_dropout(z_bin, init_from)

    net = models.build_model(
        cutout_size=cutout_size,
        n_out=n_out,
        channels=channels,
        dropout_rate=dropout,
        parallel=False,
    )

    report = models.load_pretrained(
        net,
        z_bin,
        init_from=init_from,
        root=root,
        allow_broad_reinit=allow_broad_reinit,
    )
    if report is None:
        log.info("init_from=scratch: no checkpoint loaded")
    else:
        log.info(
            "checkpoint: %.2f%% of parameters loaded, re-initialised %s, "
            "unexpected %s",
            100 * report["loaded_fraction"],
            report["reinitialised"] or "nothing",
            report["unexpected"] or "nothing",
        )

    n_frozen = models.apply_freeze(net, freeze)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(
        "freeze=%s: %d parameters frozen, %d trainable",
        freeze,
        n_frozen,
        trainable,
    )

    # Upstream wraps in DataParallel unconditionally, which breaks
    # single-GPU runs and prefixes every saved key with `module.`.
    n_gpu = torch.cuda.device_count()
    if parallel is None:
        parallel = n_gpu > 1
    if parallel and n_gpu > 1:
        net = nn.DataParallel(net)
        log.info("wrapped in DataParallel across %d GPUs", n_gpu)
    else:
        log.info("single-device run (%d GPU visible)", n_gpu)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device)
    return (
        net,
        report,
        device,
        {
            "n_frozen": n_frozen,
            "trainable": trainable,
            "dropout": dropout,
            "n_out": n_out,
            "parallel": bool(parallel and n_gpu > 1),
        },
    )


# --- 6. optimiser, loss, schedule --------------------------------------------

HEAD_PATTERNS = ("fc_loc.0", "vgg.classifier")


def build_optimizer(
    net,
    lr,
    head_lr_mult=10.0,
    momentum=0.99,
    weight_decay=1e-4,
    nesterov=False,
    patience=25,
):
    """SGD with a faster head, the aleatoric covariance loss, and a schedule.

    ``fc_loc.0`` is the one re-initialised tensor and the classifier is the
    only part whose targets changed, so both need to move faster than a
    backbone that is already close to right.
    """
    import torch
    import torch.optim as opt
    from ggt.losses import AleatoricCovLoss

    head, backbone = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        (head if any(p in name for p in HEAD_PATTERNS) else backbone).append(
            param
        )

    groups = [
        {"params": backbone, "lr": lr, "name": "backbone"},
        {"params": head, "lr": lr * head_lr_mult, "name": "head"},
    ]
    optimizer = opt.SGD(
        groups,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    log.info(
        "optimiser: %d backbone tensors @ lr %.2e, %d head tensors @ lr %.2e",
        len(backbone),
        lr,
        len(head),
        lr * head_lr_mult,
    )

    criterion = AleatoricCovLoss(num_var=len(TARGET_COLUMNS), average=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
        patience=max(1, patience // 3),
    )
    return optimizer, criterion, scheduler


# --- 7/8. run and record -----------------------------------------------------


def write_config(path, args, extra):
    """Dump everything needed to reconstruct this run from the file alone."""
    payload = {
        **{k: _jsonable(v) for k, v in vars(args).items()},
        **{k: _jsonable(v) for k, v in extra.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("wrote %s", path)
    return payload


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run(args):
    """The whole pipeline, in order."""
    from ggt.train import create_trainer

    root = args.data_root
    resolved = resolve(
        args.z_bin, args.band, args.run_name, root, args.figures_dir
    )
    manifest = check_pixel_zp(args.z_bin, args.pixel_zp, root)

    split_manifest, scaler = prepare_split(
        args.z_bin,
        args.band,
        seed=args.seed,
        force_resplit=args.force_resplit,
        root=root,
    )

    loaders, train_eval_loader, sizes, cutout_size = build_loaders(
        args.z_bin,
        args.band,
        scaler,
        seed=args.seed,
        cutout_size=args.cutout_size,
        channels=args.channels,
        repeat_dims=args.repeat_dims,
        normalize=args.normalize,
        transform=args.transform,
        expand_data=args.expand_data,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        root=root,
        limit=args.limit,
    )

    net, report, device, net_info = build_net(
        args.z_bin,
        cutout_size,
        init_from=args.init_from,
        dropout=args.dropout,
        channels=args.channels,
        freeze=args.freeze,
        allow_broad_reinit=args.allow_broad_reinit,
        root=root,
    )

    optimizer, criterion, scheduler = build_optimizer(
        net,
        args.lr,
        head_lr_mult=args.head_lr_mult,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        patience=args.patience,
    )

    ckpt = models.checkpoint_path(args.z_bin, args.init_from, root)
    # This package lives in the fork; the analysis repo is whatever encloses
    # it (or the cwd, when the fork is checked out standalone).
    fork_root = enclosing_repo(Path(__file__))
    repo_root = enclosing_repo(fork_root.parent) if fork_root else None
    if repo_root is None:
        repo_root = enclosing_repo(Path.cwd())
    fork_sha, fork_branch = git_describe(fork_root or Path.cwd())
    repo_sha, repo_branch = git_describe(repo_root or Path.cwd())
    write_config(
        resolved["config_json"],
        args,
        {
            "resolved_cutout_size": cutout_size,
            "split_sizes": sizes,
            "split_slug": split_manifest["slug"],
            "split_hashes": split_manifest["hashes"],
            "scaler_mean": list(scaler.mean_),
            "scaler_scale": list(scaler.scale_),
            "target_columns": TARGET_COLUMNS,
            "checkpoint_path": str(ckpt) if ckpt else None,
            "checkpoint_report": report,
            "cache_pixel_zp": manifest.get("pixel_zp"),
            "cache_crop_px": manifest.get("crop_px"),
            "device": device,
            "figures_dir": str(resolved["figures_dir"]),
            "repo_git_sha": repo_sha,
            "repo_git_branch": repo_branch,
            "fork_git_sha": fork_sha,
            "fork_git_branch": fork_branch,
            **net_info,
        },
    )

    if args.mlflow:
        os.environ.setdefault(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{resolved['run_dir'] / 'mlflow.db'}",
        )

    trainer, _evaluator = create_trainer(
        net,
        optimizer,
        criterion,
        loaders,
        device,
        eval_splits=("train", "devel"),
        eval_train=True,
        train_eval_loader=train_eval_loader,
        train_eval_every=args.train_eval_every,
        checkpoint_dir=resolved["run_dir"],
        patience=args.patience,
        lr_scheduler=scheduler,
        metrics_csv=resolved["metrics_csv"],
        log_file=resolved["train_log"],
        target_names=TARGET_COLUMNS,
        use_mlflow=args.mlflow,
    )

    log.info(
        "training for up to %d epochs -> %s", args.epochs, resolved["run_dir"]
    )
    trainer.run(loaders["train"], max_epochs=args.epochs)
    log.info(
        "done: best devel loss %.4f at epoch %s",
        trainer.state.best_devel_loss,
        trainer.state.best_epoch,
    )

    if not args.no_figures:
        try:
            from ggt.visualization import training_figures

            training_figures.make_all(
                resolved["run_dir"], resolved["figures_dir"]
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("figures failed (%s); run_dir is still complete", exc)

    return resolved["run_dir"]


# --- CLI ---------------------------------------------------------------------


def parse_pixel_zp(value):
    if value is None or str(value).lower() in ("none", "", "null"):
        return None
    return float(value)
# --- CLI ---------------------------------------------------------------


@click.command()
@click.option("--z-bin", type=int, required=True)
@click.option(
    "--band",
    type=click.Choice(euclid.BANDS, case_sensitive=False),
    required=True,
)
@click.option("--run-name", type=str, required=True)
@click.option("--data-root", type=str, default=None)
@click.option("--seed", type=int, default=42)
@click.option(
    "--cutout-size",
    type=int,
    default=None,
    help="Side of the image the network sees. Defaults to the bin's "
    "target_crop_px; must not exceed the cache's crop_px.",
)
@click.option(
    "--pixel-zp",
    type=str,
    default="none",
    help="'none' for native survey units, or a zeropoint. MUST match how "
    "the cache was built; the run refuses to start otherwise.",
)
@click.option("--channels", type=int, default=3)
@click.option("--repeat-dims/--no-repeat-dims", default=True)
@click.option("--normalize/--no-normalize", default=True)
@click.option(
    "--transform/--no-transform",
    default=True,
    help="Augment the train split with flips and rotations.",
)
@click.option(
    "--init-from",
    type=click.Choice(["real", "sim", "scratch"], case_sensitive=False),
    default="real",
    help="Which published checkpoint family to start from.",
)
@click.option(
    "--freeze",
    type=click.Choice(list(models.FREEZE_SPECS), case_sensitive=False),
    default="vgg_features_early",
)
@click.option(
    "--allow-broad-reinit",
    is_flag=True,
    default=False,
    help="Permit tensors other than fc_loc.0.weight to be re-initialised. "
    "Almost always a mistake: it means the transfer is not doing what you "
    "think it is.",
)
@click.option(
    "--head-lr-mult",
    type=float,
    default=10.0,
    help="LR multiplier for fc_loc.0 and the classifier.",
)
@click.option(
    "--lr",
    type=float,
    default=5e-7,
    help="Backbone learning rate. aleatoric_cov exponentiates its variance "
    "terms; anything above ~1e-6 is liable to diverge.",
)
@click.option("--momentum", type=float, default=0.99)
@click.option("--weight-decay", type=float, default=1e-4)
@click.option("--nesterov/--no-nesterov", default=False)
@click.option(
    "--dropout",
    type=float,
    default=None,
    help="Defaults to the checkpoint's published rate.",
)
@click.option("--batch-size", type=int, default=16)
@click.option("--epochs", type=int, default=200)
@click.option(
    "--patience",
    type=int,
    default=25,
    help="Early-stopping patience. ReduceLROnPlateau uses a third of it.",
)
@click.option(
    "--expand-data",
    type=int,
    default=4,
    help="Augmentation factor, train split only.",
)
@click.option("--n-workers", type=int, default=8)
@click.option("--train-eval-every", type=int, default=1)
@click.option(
    "--force-resplit",
    is_flag=True,
    default=False,
    help="Refused on an existing seed; pick a new --seed instead.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Truncate every split. For overfit and smoke tests.",
)
@click.option("--mlflow", is_flag=True, default=False)
@click.option("--no-figures", is_flag=True, default=False)
@click.option(
    "--figures-dir",
    type=str,
    default=None,
    help="Where figures go; defaults to training_eval_figs/ inside the "
    "run directory.",
)
def main(**kwargs):
    """Fine-tune a GaMPEN model on one (z_bin, band) dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        stream=sys.stdout,
    )
    kwargs["pixel_zp"] = parse_pixel_zp(kwargs["pixel_zp"])
    kwargs["band"] = kwargs["band"].upper()
    run(SimpleNamespace(**kwargs))


if __name__ == "__main__":
    main()
