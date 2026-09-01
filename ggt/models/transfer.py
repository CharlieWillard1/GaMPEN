# -*- coding: utf-8 -*-
"""Transfer learning: build a model, load a checkpoint into it, freeze it.

The survey-specific half -- which published checkpoint initialises which
redshift bin -- lives in `ggt.surveys.euclid`. What is here is generic:
constructing the network without the wasted ImageNet download, loading
weights across an input-size change, and freezing named parameter groups.

Measured against the real HSC checkpoints: every Euclid bin re-initialises
exactly one tensor, `fc_loc.0.weight`, transferring 83.8-97.7% of the model
by parameter count. `fc_loc.0.bias` is [32] at any input size, so it
transfers too -- one tensor, not two.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ggt.data import layout
from ggt.surveys import euclid

log = logging.getLogger("ggt.models.transfer")

# The only parameter whose shape depends on cutout_size -- see
# `load_checkpoint_tolerant`. With torchvision VGG16's AdaptiveAvgPool2d,
# the STN's single-scalar zoom head is the sole size-dependent tensor, and
# that is what makes this whole approach work.
SIZE_DEPENDENT = {"fc_loc.0.weight"}

# Freezing specs are computed from the live model rather than hardcoded,
# because enabling dropout inserts Dropout layers into vgg.features and
# shifts every Conv2d index (conv weights land at 0, 3, 7, 10, 14, 17, 20,
# 24, 27, 30, 34, 37, 40 -- not 0..12).
#
# Parameter budget at cutout_size=400 (measured), which is why the obvious
# spec is not the useful one:
#     vgg.classifier  119,582,729   74%   <- the bulk of the model
#     fc_loc           26,001,473   16%   <- the re-initialised STN head
#     vgg.features     14,714,688    9%
#     localization        521,056    0.3%
# Freezing the convolutional features therefore frees only ~9% of the
# parameters. If the goal is to fit ~10k labels without overfitting, the
# classifier is the target.
FREEZE_SPECS = (
    "none",
    "stn",
    "vgg_features",
    "vgg_features_early",
    "vgg_classifier",
    "all_but_head",
)


def _conv_prefixes(model, first_n=None):
    """Prefixes of vgg.features conv weight tensors, in depth order."""
    names = [
        n
        for n, _ in model.named_parameters()
        if n.startswith("vgg.features") and n.endswith(".weight")
    ]
    if first_n is not None:
        names = names[:first_n]
    return [n[: -len("weight")] for n in names]


def freeze_patterns(model, spec: str):
    """Name patterns to freeze for a spec, resolved against this model."""
    if spec == "none":
        return []
    if spec == "stn":
        return ["localization", "fc_loc"]
    if spec == "vgg_features":
        return ["vgg.features"]
    if spec == "vgg_features_early":
        # first 7 of 13 conv layers, i.e. roughly VGG blocks 1-3
        return _conv_prefixes(model, 7)
    if spec == "vgg_classifier":
        return ["vgg.classifier"]
    if spec == "all_but_head":
        # all but the re-initialised STN head and the output layer
        return [
            "localization",
            "vgg.features",
            "vgg.classifier.0",
            "vgg.classifier.3",
        ]
    raise ValueError(f"freeze must be one of {FREEZE_SPECS}, got {spec!r}")


def checkpoint_path(z_bin: int, init_from: str = "real", root=None):
    """Resolve the published checkpoint that initialises this bin."""
    if init_from == "scratch":
        return None
    idx = {"real": 0, "sim": 1}.get(init_from)
    if idx is None:
        raise ValueError(
            f"init_from must be real|sim|scratch, got {init_from!r}"
        )
    name = euclid.CHECKPOINT_FOR_BIN[z_bin][idx]
    p = (
        layout.pretrained_dir(root)
        / name
        / "trained_model"
        / euclid.CHECKPOINT_FILE[name]
    )
    if not p.exists():
        raise SystemExit(f"{p} not found; run scripts/fetch_hsc_models.sh")
    return p


def default_dropout(z_bin: int, init_from: str = "real") -> float:
    idx = {"real": 0, "sim": 1}.get(init_from, 0)
    return euclid.DROPOUT_FOR[euclid.CHECKPOINT_FOR_BIN[z_bin][idx]]


def build_model(
    cutout_size,
    n_out=9,
    channels=3,
    dropout_rate=None,
    parallel=False,
    model_type=euclid.MODEL_TYPE,
):
    """Construct the model without the wasted ImageNet download.

    `pretrained=False` because the VGG weights are immediately overwritten
    with a survey checkpoint; the 528 MB ImageNet download is pure waste in
    that case.
    """
    import torch.nn as nn

    from ggt.models import model_factory
    from ggt.utils import specify_dropout_rate

    cls = model_factory(model_type)
    model = cls(
        cutout_size=cutout_size,
        channels=channels,
        n_out=n_out,
        pretrained=False,
        dropout="True",
    )
    if dropout_rate is not None:
        specify_dropout_rate(model, dropout_rate)
    if parallel:
        model = nn.DataParallel(model)
    return model


def load_pretrained(
    model,
    z_bin,
    init_from="real",
    root=None,
    allow_broad_reinit=False,
    checkpoint=None,
):
    """Load a checkpoint, allowing only `fc_loc.0.weight` to reset.

    Anything else being re-initialised means the transfer is not doing what
    we think, so it raises rather than training silently from scratch.

    `checkpoint` overrides the published-checkpoint lookup with an explicit
    path -- typically a `best.pt` from an earlier run of our own, to
    continue training under different settings. Such a checkpoint has the
    same geometry as the model being built, so it should load at 100% with
    nothing re-initialised; if it does not, the two runs disagree about
    something, and that is worth understanding before continuing.
    """
    import torch

    from ggt.utils import load_checkpoint_tolerant

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise SystemExit(f"--init-checkpoint {path} does not exist")
    else:
        path = checkpoint_path(z_bin, init_from, root)
    if path is None:
        log.info("init_from=scratch: no checkpoint loaded")
        return None

    sd = torch.load(path, map_location="cpu", weights_only=True)
    report = load_checkpoint_tolerant(
        model,
        sd,
        allow_reinit=None if allow_broad_reinit else SIZE_DEPENDENT,
    )
    log.info("loaded %s", path.name)
    log.info(
        "  transferred %.2f%% of parameters (%d tensors)",
        report["loaded_fraction"] * 100,
        len(report["loaded"]),
    )
    log.info("  re-initialised: %s", report["reinitialised"] or "nothing")
    if report["unexpected"]:
        log.warning("  unexpected keys: %s", report["unexpected"])
    return report


def apply_freeze(model, spec="none"):
    """Freeze a named group. Returns the number of parameters frozen."""
    from ggt.utils import set_requires_grad

    patterns = freeze_patterns(model, spec)
    total = sum(p.numel() for p in model.parameters())
    if not patterns:
        log.info("freeze 'none': all %s parameters trainable", f"{total:,}")
        return 0
    n = set_requires_grad(model, patterns, requires_grad=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "freeze '%s': froze %s of %s parameters (%.1f%%); %s trainable",
        spec,
        f"{n:,}",
        f"{total:,}",
        100 * n / total,
        f"{trainable:,}",
    )
    return n
