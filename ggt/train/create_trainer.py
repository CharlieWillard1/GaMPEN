import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, Loss

from ggt.metrics import ElementwiseMae
from ggt.losses import AleatoricLoss, AleatoricCovLoss
from ggt.utils import (
    metric_output_transform_al_loss,
    metric_output_transform_al_cov_loss,
)


def collect_metrics(evaluator, loader):
    """Run `evaluator` over `loader` and return a plain-Python copy.

    `evaluator.state.metrics` is overwritten by the next `run()`, and
    `elementwise_mae` is a tensor rather than a float. Anything that wants to
    hold two evaluations at once -- which is exactly what comparing train
    against devel requires -- has to copy the values out first.
    """
    evaluator.run(loader)
    out = {}
    for name, value in evaluator.state.metrics.items():
        out[name] = value.tolist() if hasattr(value, "tolist") else value
    return out


def log_metrics_to_mlflow(prefix, metrics, step):
    """Flatten `elementwise_mae` into one metric per target, as before.

    `mlflow` is imported here rather than at module scope so that
    `import ggt.train` works without it. It is a heavyweight dependency to
    require of anyone who only wants to build a trainer.
    """
    import mlflow

    for name, value in metrics.items():
        if isinstance(value, list):
            for i, val in enumerate(value):
                mlflow.log_metric(f"{prefix}-{name}-{i}", val, step)
        else:
            mlflow.log_metric(f"{prefix}-{name}", value, step)


def metrics_row(epoch, lr, wall_seconds, train, devel, target_names):
    """One flat `metrics.csv` row from a train and a devel metric dict."""
    row = {"epoch": epoch, "lr": lr, "wall_seconds": round(wall_seconds, 3)}
    for prefix, m in (("train", train), ("devel", devel)):
        for key in ("loss", "mae", "mse"):
            row[f"{prefix}_{key}"] = m.get(key) if m else None
        per_target = (m or {}).get("elementwise_mae") or []
        for name, value in zip(target_names, per_target):
            row[f"{prefix}_mae_{name}"] = value
        for name in target_names[len(per_target) :]:
            row[f"{prefix}_mae_{name}"] = None
    return row


def metrics_fieldnames(target_names):
    """Column order for `metrics.csv`."""
    names = ["epoch", "lr", "wall_seconds"]
    for key in ("loss", "mae", "mse"):
        names += [f"train_{key}", f"devel_{key}"]
    for prefix in ("train", "devel"):
        names += [f"{prefix}_mae_{n}" for n in target_names]
    return names


def save_state_dict(model, path):
    """Write `model`'s state dict to `path` atomically.

    Via a temporary file and `os.replace`, so that killing a run part-way
    through a write leaves the previous checkpoint intact rather than a
    truncated file. `last.pt` is rewritten every epoch, which is exactly when
    an impatient Ctrl-C is most likely to land.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
    return path


def create_trainer(
    model,
    optimizer,
    criterion,
    loaders,
    device,
    eval_splits=None,
    eval_train=False,
    train_eval_loader=None,
    train_eval_every=1,
    checkpoint_dir=None,
    patience=None,
    lr_scheduler=None,
    metrics_csv=None,
    log_file=None,
    target_names=None,
    use_mlflow=True,
):
    """Set up Ignite trainer and evaluator.

    Returns `(trainer, evaluator)`. The evaluator used to be reachable only
    from inside this function's closures, which made it impossible for a
    caller to attach checkpointing or early stopping to the devel metrics.

    Parameters
    ----------
    eval_splits : sequence of str, optional
        Which of `loaders` to evaluate at the start and end of training.
        Defaults to all of them, which is the historical behaviour. Pass
        `("train", "devel")` to leave the test split untouched by a training
        run.
    eval_train : bool
        Evaluate the training set every epoch as well as the devel set. Off
        by default, matching the historical behaviour in which train metrics
        were recorded only at epoch 0 and at the end -- which is precisely
        why the resulting curves could not show overfitting.
    train_eval_loader : torch.utils.data.DataLoader, optional
        Loader used for that per-epoch train evaluation. Supply an
        un-augmented, un-expanded loader over the training set: a training
        loss measured through random flips and rotations is not the same
        quantity as the devel loss and cannot be plotted against it.
        Defaults to `loaders["train"]`.
    train_eval_every : int
        Evaluate the training set every N epochs. Doing it every epoch
        roughly doubles the per-epoch cost.
    checkpoint_dir : path-like, optional
        Write `best.pt` (lowest devel loss so far) and `last.pt` (every
        epoch) here. Without this only the final model is saved, and the
        final model is generally the overfitted one.
    patience : int, optional
        Stop early after this many consecutive epochs without an improvement
        in devel loss.
    lr_scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau, optional
        Stepped with the devel loss after each epoch's evaluation.
    metrics_csv : path-like, optional
        Append one row per epoch here, flushed immediately -- the point of
        the file is to be readable while the run is still going.
    log_file : path-like, optional
        Append one human-readable line per epoch here, for `tail -f`.
    target_names : sequence of str, optional
        Names for the per-target `elementwise_mae` columns. Defaults to
        positional indices.
    use_mlflow : bool
        Log to MLflow as well. On MLflow 3.x the default `file://` store
        silently records nothing -- no `meta.yaml`, no metrics, no params,
        and `search_runs` then returns zero runs without ever raising. Use a
        sqlite tracking URI if you turn this on, and treat the flat files as
        the source of truth either way.

    Notes
    -----
    After training, `trainer.state.best_devel_loss` and
    `trainer.state.best_epoch` hold the best score and the epoch it came from.
    """
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    if isinstance(criterion, AleatoricLoss):
        output_transform = metric_output_transform_al_loss
    elif isinstance(criterion, AleatoricCovLoss):
        output_transform = metric_output_transform_al_cov_loss
    else:

        def output_transform(x):
            return x

    metrics = {
        "mae": MeanAbsoluteError(output_transform=output_transform),
        "elementwise_mae": ElementwiseMae(output_transform=output_transform),
        "mse": MeanSquaredError(output_transform=output_transform),
        "loss": Loss(criterion),
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    if eval_splits is None:
        eval_splits = tuple(loaders.keys())
    if train_eval_loader is None:
        train_eval_loader = loaders.get("train")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # A run killed mid-write leaves its partial file behind. These are
        # full-sized -- 643 MB for a 400 px model -- so sweep them rather
        # than letting each crash cost another checkpoint's worth of disk.
        for stale in checkpoint_dir.glob("*.pt.tmp"):
            stale.unlink()
            logging.warning("removed stale partial checkpoint %s", stale)

    if target_names is None:
        target_names = []
    target_names = list(target_names)

    fieldnames = metrics_fieldnames(target_names)
    if metrics_csv is not None:
        metrics_csv = Path(metrics_csv)
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_csv.exists():
            with open(metrics_csv, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    # Mutable, because the epoch handler below is a closure over them.
    best = {"loss": None, "epoch": None, "since_improved": 0}
    clock = {"last": time.time()}

    def append_row(row):
        with open(metrics_csv, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writerow(row)

    def append_log_line(row):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total = trainer.state.max_epochs
        train_loss = row["train_loss"]
        train_str = "  n/a " if train_loss is None else f"{train_loss:.4f}"
        line = (
            f"[{stamp}] epoch {row['epoch']:4d}/{total}  "
            f"lr {row['lr']:.1e}  "
            f"train {train_str}  devel {row['devel_loss']:.4f}  "
            f"(best {best['loss']:.4f} @ {best['epoch']})  "
            f"{row['wall_seconds']:.0f}s"
        )
        with open(log_file, "a") as fh:
            fh.write(line + "\n")

    def evaluate_bookends(step):
        for split in eval_splits:
            loader = loaders.get(split)
            if loader is None:
                continue
            metrics_ = collect_metrics(evaluator, loader)
            if use_mlflow:
                log_metrics_to_mlflow(split, metrics_, step)

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        evaluate_bookends(0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_results(trainer):
        epoch = trainer.state.epoch

        train = None
        if eval_train and epoch % train_eval_every == 0:
            train = collect_metrics(evaluator, train_eval_loader)
            if use_mlflow:
                log_metrics_to_mlflow("train", train, epoch)

        devel = collect_metrics(evaluator, loaders["devel"])
        if use_mlflow:
            log_metrics_to_mlflow("devel", devel, epoch)

        devel_loss = devel["loss"]

        # Step the schedule before early stopping, so a plateau gets its
        # chance at a lower learning rate before the run is abandoned.
        if lr_scheduler is not None:
            lr_scheduler.step(devel_loss)

        improved = best["loss"] is None or devel_loss < best["loss"]
        if improved:
            best["loss"] = devel_loss
            best["epoch"] = epoch
            best["since_improved"] = 0
        else:
            best["since_improved"] += 1

        trainer.state.best_devel_loss = best["loss"]
        trainer.state.best_epoch = best["epoch"]

        now = time.time()
        row = metrics_row(
            epoch,
            optimizer.param_groups[0]["lr"],
            now - clock["last"],
            train,
            devel,
            target_names,
        )
        clock["last"] = now
        if metrics_csv is not None:
            append_row(row)
        if log_file is not None:
            append_log_line(row)

        if checkpoint_dir is not None:
            save_state_dict(model, checkpoint_dir / "last.pt")
            if improved:
                save_state_dict(model, checkpoint_dir / "best.pt")

        if patience is not None and best["since_improved"] >= patience:
            trainer.terminate()

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        evaluate_bookends(trainer.state.epoch)

    return trainer, evaluator
