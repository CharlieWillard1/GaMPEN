import torch


def get_output_shape(model, image_dim):
    """Get output shape of a PyTorch model or layer"""
    return model(torch.rand(*(image_dim))).data.shape


def enable_dropout(model):
    """Enable random dropout during inference. From StackOverflow #63397197"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def specify_dropout_rate(model, rate):
    """Specify the dropout rate of all layers"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.p = rate


def strip_module_prefix(state_dict):
    """Remove the ``module.`` prefix that ``nn.DataParallel`` adds.

    ``save_trained_model`` stores the state_dict of the *wrapped* model, so
    every published checkpoint is prefixed. Loading one into an unwrapped
    model otherwise fails on every key.
    """
    if not any(k.startswith("module.") for k in state_dict):
        return dict(state_dict)
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def load_checkpoint_tolerant(
    model, state_dict, allow_reinit=None, strict_shapes=True
):
    """Load a checkpoint, skipping tensors whose shape does not match.

    Transfer learning across input sizes is a first-class use of these
    models: only the STN's localization head depends on ``cutout_size``,
    so a checkpoint trained at one size is otherwise fully reusable at
    another. ``load_state_dict`` cannot express that -- it is
    all-or-nothing on shapes -- so this drops the mismatched tensors and
    reports exactly what happened.

    Args:
        model: target module, wrapped or not.
        state_dict: checkpoint contents.
        allow_reinit: iterable of parameter names permitted to be
            re-initialised; anything else that fails to load raises.
            ``None`` disables the check. This exists because a silent
            broad mismatch -- a wrong ``cutout_size``, a torchvision
            change -- otherwise degrades into a from-scratch run that
            still trains and still produces plausible loss curves.
        strict_shapes: raise if the checkpoint has keys the model lacks.

    Returns:
        dict with ``loaded``, ``reinitialised``, ``mismatched_shapes``,
        ``unexpected`` and ``loaded_fraction`` (of parameters, by count).
    """
    sd = strip_module_prefix(state_dict)
    target = model.state_dict()
    is_wrapped = any(k.startswith("module.") for k in target)
    if is_wrapped:
        target = strip_module_prefix(target)

    loadable, mismatched = {}, []
    for k, v in sd.items():
        if k in target and tuple(v.shape) == tuple(target[k].shape):
            loadable[k] = v
        elif k in target:
            mismatched.append(k)
    unexpected = sorted(set(sd) - set(target))
    reinit = sorted(set(target) - set(loadable))

    if strict_shapes and unexpected:
        raise ValueError(
            "checkpoint has %d keys the model does not: %s"
            % (len(unexpected), unexpected[:5])
        )
    if allow_reinit is not None:
        bad = sorted(set(reinit) - set(allow_reinit))
        if bad:
            raise ValueError(
                "%d tensors would be re-initialised but were not "
                "permitted: %s. Expected only %s. A broad mismatch "
                "usually means the model was built with the wrong "
                "cutout_size/channels, and would silently train from "
                "scratch." % (len(bad), bad[:8], sorted(allow_reinit))
            )

    to_load = (
        {"module." + k: v for k, v in loadable.items()}
        if is_wrapped
        else loadable
    )
    model.load_state_dict(to_load, strict=False)

    n_loaded = sum(v.numel() for v in loadable.values())
    n_total = sum(v.numel() for v in target.values())
    return {
        "loaded": sorted(loadable),
        "reinitialised": reinit,
        "mismatched_shapes": sorted(mismatched),
        "unexpected": unexpected,
        "loaded_fraction": n_loaded / n_total if n_total else 0.0,
    }


def set_requires_grad(model, patterns, requires_grad=False):
    """Freeze or unfreeze parameters whose name contains any pattern.

    Returns the number of parameters affected, by count, so a caller can
    log something more useful than "froze some layers".
    """
    n = 0
    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            p.requires_grad = requires_grad
            n += p.numel()
    return n
