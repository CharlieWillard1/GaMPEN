import mlflow

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
    """Flatten `elementwise_mae` into one metric per target, as before."""
    for name, value in metrics.items():
        if isinstance(value, list):
            for i, val in enumerate(value):
                mlflow.log_metric(f"{prefix}-{name}-{i}", val, step)
        else:
            mlflow.log_metric(f"{prefix}-{name}", value, step)


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

    def evaluate_bookends(step):
        for split in eval_splits:
            loader = loaders.get(split)
            if loader is None:
                continue
            log_metrics_to_mlflow(
                split, collect_metrics(evaluator, loader), step
            )

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        evaluate_bookends(0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_results(trainer):
        epoch = trainer.state.epoch

        if eval_train and epoch % train_eval_every == 0:
            log_metrics_to_mlflow(
                "train", collect_metrics(evaluator, train_eval_loader), epoch
            )

        log_metrics_to_mlflow(
            "devel", collect_metrics(evaluator, loaders["devel"]), epoch
        )

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        evaluate_bookends(trainer.state.epoch)

    return trainer, evaluator
