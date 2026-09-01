# -*- coding: utf-8 -*-
"""The centre-crop convention, defined once.

Some upstream survey pipelines document `crop_size` as "assumed even",
computing `half = crop // 2` and slicing symmetrically about `image // 2`,
which for an ODD crop silently returns `crop - 1` pixels.

Production Euclid crops (400/320/240/200/160) are all even, so that
assumption holds there. But the HSC checkpoints use 239, 143 and 96 -- two
of which are odd -- and the native-vs-resampled ablation crops to 239. A
silent off-by-one there surfaces much later as an opaque `X.view()` size
error inside `FITSDataset.__getitem__`.

So this uses the standard formula `start = (image - crop) // 2`, which is
identical for every even size and correct for odd ones too. The Euclid
repo's `tests/test_cache_fidelity.py` pins both halves of that claim.
"""

from __future__ import annotations

import numpy as np


def crop_center(arr, crop_size, image_size):
    """Centre-crop the last two axes to `crop_size` x `crop_size`.

    Returns a view when possible. `crop_size >= image_size` returns the
    array unchanged.
    """
    crop_size, image_size = int(crop_size), int(image_size)
    if crop_size >= image_size:
        return arr
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    start = (image_size - crop_size) // 2
    stop = start + crop_size
    out = np.asarray(arr)[..., start:stop, start:stop]
    assert out.shape[-2:] == (crop_size, crop_size), (
        out.shape,
        crop_size,
        image_size,
    )
    return out
