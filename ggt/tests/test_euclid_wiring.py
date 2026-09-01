"""Wiring tests for the training launcher (Plan C).

These are the checks that catch a mis-wiring before a multi-hour GPU run
rather than after it. The ones that need real data skip when the cache is
absent, in the same style as `test_cache_fidelity.py`; the rest are pure
logic and always run.
"""

import json

import numpy as np
import pytest

from ggt.data import layout, splits
from ggt.models import transfer as model
from ggt.surveys import euclid as config
from ggt.train import train

Z_BIN = 0
BAND = "VIS"

needs_cache = pytest.mark.skipif(
    not layout.cache_path(Z_BIN).exists(), reason="cache not built"
)
needs_info = pytest.mark.skipif(
    not layout.info_csv(Z_BIN, BAND).exists(), reason="info.csv not built"
)


# --- target order ------------------------------------------------------------

def test_target_order_matches_pretrained_head():
    """The output head's order is fixed by the HSC checkpoint."""
    assert train.TARGET_COLUMNS == [
        "custom_logit_bt",
        "ln_R_e_asec",
        "ln_total_flux_adus",
    ]
    assert tuple(train.TARGET_COLUMNS) == config.TARGET_COLUMNS


def test_n_out_matches_aleatoric_cov():
    """aleatoric_cov needs (3n + n^2)/2 outputs for n targets."""
    n = len(train.TARGET_COLUMNS)
    assert int((3 * n + n ** 2) / 2) == 9


# --- the pixel-zeropoint guard -----------------------------------------------

def _write_manifest(tmp_path, pixel_zp):
    bin_dir = tmp_path / "gampen_data" / config.bin_dirname(Z_BIN)
    bin_dir.mkdir(parents=True)
    (bin_dir / "cache_manifest.json").write_text(
        json.dumps({"pixel_zp": pixel_zp, "crop_px": 400})
    )
    return tmp_path


def test_pixel_zp_guard_accepts_a_match(tmp_path):
    root = _write_manifest(tmp_path, None)
    assert train.check_pixel_zp(Z_BIN, None, root=root)["crop_px"] == 400

    root2 = _write_manifest(tmp_path / "b", 27.0)
    assert train.check_pixel_zp(Z_BIN, 27.0, root=root2)["pixel_zp"] == 27.0


def test_pixel_zp_guard_rejects_a_mismatch(tmp_path):
    """Both values must appear in the message -- that is the whole point."""
    root = _write_manifest(tmp_path, None)
    with pytest.raises(SystemExit) as exc:
        train.check_pixel_zp(Z_BIN, 27.0, root=root)
    message = str(exc.value)
    assert "27" in message and "None" in message


def test_pixel_zp_guard_rejects_the_reverse_mismatch(tmp_path):
    root = _write_manifest(tmp_path, 27.0)
    with pytest.raises(SystemExit):
        train.check_pixel_zp(Z_BIN, None, root=root)


def test_parse_pixel_zp():
    assert train.parse_pixel_zp("none") is None
    assert train.parse_pixel_zp("NONE") is None
    assert train.parse_pixel_zp("") is None
    assert train.parse_pixel_zp("27") == 27.0


# --- missing data fails with a usable message --------------------------------

def test_resolve_names_the_build_commands(tmp_path):
    with pytest.raises(SystemExit) as exc:
        train.resolve(Z_BIN, BAND, "run", root=tmp_path)
    message = str(exc.value)
    assert "build_cache" in message and "build_info" in message


def test_resolve_rejects_an_unknown_band(tmp_path):
    with pytest.raises(SystemExit):
        train.resolve(Z_BIN, "SDSS-r", "run", root=tmp_path)


# --- CLI defaults hold the values the plan settled on ------------------------

def _cli_params():
    return {p.name: p for p in train.main.params}


def test_cli_defaults():
    d = {name: p.default for name, p in _cli_params().items()}
    assert d["lr"] == 5e-7  # >1e-6 diverges on aleatoric_cov
    assert d["freeze"] == "vgg_features_early"
    assert d["momentum"] == 0.99
    assert d["weight_decay"] == 1e-4
    assert d["batch_size"] == 16
    assert d["patience"] == 25
    assert d["expand_data"] == 4
    assert d["head_lr_mult"] == 10.0
    assert d["init_from"] == "real"
    assert d["transform"] is True
    assert d["pixel_zp"] == "none"


def test_freeze_choices_come_from_the_model_module():
    opt = _cli_params()["freeze"]
    assert tuple(opt.type.choices) == model.FREEZE_SPECS


def test_required_cli_options():
    """A run must always name its bin, band and run directory."""
    required = {n for n, p in _cli_params().items() if p.required}
    assert required == {"z_bin", "band", "run_name"}


# --- checkpoint surgery ------------------------------------------------------

@pytest.mark.skipif(
    not config.CHECKPOINT_FILE
    or not (layout.pretrained_dir() / "g_0_025").exists(),
    reason="HSC checkpoints not fetched",
)
def test_only_fc_loc_0_weight_is_reinitialised():
    """The premise the whole transfer-learning approach rests on.

    `fc_loc.0.weight` is the only input-size-dependent tensor in the model.
    The bias is [32] at any input size, so it transfers too -- one tensor,
    not two.
    """
    net = model.build_model(cutout_size=400, n_out=9, channels=3)
    report = model.load_pretrained(net, Z_BIN, init_from="real")
    assert report["reinitialised"] == ["fc_loc.0.weight"]
    assert report["unexpected"] == []
    assert report["loaded_fraction"] > 0.83


# --- freezing and parameter groups -------------------------------------------

def test_freeze_reduces_the_trainable_count():
    net = model.build_model(cutout_size=96, n_out=9, channels=3)
    before = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = model.apply_freeze(net, "vgg_features_early")
    after = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert frozen > 0
    assert after == before - frozen


def test_head_group_gets_the_multiplied_lr():
    """fc_loc.0 and the classifier must land in the faster group."""
    net = model.build_model(cutout_size=96, n_out=9, channels=3)
    optimizer, criterion, scheduler = train.build_optimizer(
        net, lr=5e-7, head_lr_mult=10.0, patience=9
    )
    groups = {g["name"]: g for g in optimizer.param_groups}
    assert set(groups) == {"backbone", "head"}
    assert np.isclose(groups["head"]["lr"], 10 * groups["backbone"]["lr"])
    assert groups["head"]["params"], "head group must not be empty"
    assert groups["backbone"]["params"], "backbone group must not be empty"
    # ReduceLROnPlateau patience is a third of the early-stopping patience.
    assert scheduler.patience == 3


def test_frozen_parameters_stay_out_of_the_optimizer():
    net = model.build_model(cutout_size=96, n_out=9, channels=3)
    model.apply_freeze(net, "all_but_head")
    optimizer, _, _ = train.build_optimizer(net, lr=5e-7)
    in_optimizer = {
        id(p) for g in optimizer.param_groups for p in g["params"]
    }
    for _, param in net.named_parameters():
        if not param.requires_grad:
            assert id(param) not in in_optimizer


# --- the metrics.csv contract ------------------------------------------------

def test_metrics_csv_columns_are_the_agreed_fifteen():
    from ggt.train.create_trainer import metrics_fieldnames

    assert metrics_fieldnames(train.TARGET_COLUMNS) == [
        "epoch", "lr", "wall_seconds",
        "train_loss", "devel_loss",
        "train_mae", "devel_mae",
        "train_mse", "devel_mse",
        "train_mae_custom_logit_bt",
        "train_mae_ln_R_e_asec",
        "train_mae_ln_total_flux_adus",
        "devel_mae_custom_logit_bt",
        "devel_mae_ln_R_e_asec",
        "devel_mae_ln_total_flux_adus",
    ]


def test_metrics_row_tolerates_a_skipped_train_evaluation():
    """With --train-eval-every N, train columns are empty elsewhere."""
    from ggt.train.create_trainer import metrics_row

    devel = {"loss": 1.0, "mae": 2.0, "mse": 3.0,
             "elementwise_mae": [0.1, 0.2, 0.3]}
    row = metrics_row(7, 5e-7, 12.0, None, devel, train.TARGET_COLUMNS)
    assert row["epoch"] == 7
    assert row["train_loss"] is None
    assert row["train_mae_ln_R_e_asec"] is None
    assert row["devel_loss"] == 1.0
    assert row["devel_mae_ln_total_flux_adus"] == 0.3


def test_atomic_save_leaves_no_temporary_file(tmp_path):
    import torch.nn as nn

    from ggt.train.create_trainer import save_state_dict

    out = save_state_dict(nn.Linear(2, 2), tmp_path / "best.pt")
    assert out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_stale_partial_checkpoints_are_swept(tmp_path):
    """A run killed mid-write strands a full-sized .pt.tmp; sweep it.

    Regression test: a SIGKILL during a checkpoint write left a 643 MB
    `best.pt.tmp` behind, and nothing ever removed it.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from ggt.train import create_trainer

    stale = tmp_path / "best.pt.tmp"
    stale.write_bytes(b"partial")
    keep = tmp_path / "best.pt"
    keep.write_bytes(b"real")

    ds = TensorDataset(torch.randn(4, 3), torch.randn(4, 1))
    loaders = {"train": DataLoader(ds, batch_size=2),
               "devel": DataLoader(ds, batch_size=2)}
    net = nn.Linear(3, 1)
    create_trainer(
        net, torch.optim.SGD(net.parameters(), lr=0.0), nn.MSELoss(),
        loaders, "cpu", checkpoint_dir=tmp_path, use_mlflow=False,
    )

    assert not stale.exists(), "stale .pt.tmp was not swept"
    assert keep.exists(), "the real checkpoint must not be touched"


# --- real-data checks --------------------------------------------------------

def test_reuse_rejects_a_split_that_predates_a_rebuilt_info_csv(tmp_path):
    """A stale split hashes fine against itself and is completely wrong.

    Regression test: rebuilding info.csv (debug data -> production, or a
    retuned cut) leaves the previous split self-consistent, so the hash check
    passes and the run silently trains on the old, much smaller subset.
    """
    import pandas as pd

    band_dir = tmp_path / "gampen_data" / config.bin_dirname(Z_BIN) / BAND
    (band_dir / "splits").mkdir(parents=True)

    def frame(ids):
        n = len(ids)
        return pd.DataFrame({
            "object_id": ids,
            "bt": np.linspace(0.1, 0.9, n),
            "custom_logit_bt": np.linspace(-1, 1, n),
            "ln_R_e_asec": np.linspace(-1, 1, n),
            "ln_total_flux_adus": np.linspace(5, 7, n),
        })

    # A split built when info.csv held ids 0..19.
    frame(list(range(20))).to_csv(band_dir / "info.csv", index=False)
    splits.build(Z_BIN, BAND, seed=42, root=tmp_path)
    assert splits.build(
        Z_BIN, BAND, seed=42, root=tmp_path
    ), "reuse should work"

    # info.csv is rebuilt with different sources; the split is now stale.
    frame(list(range(100, 160))).to_csv(band_dir / "info.csv", index=False)
    with pytest.raises(SystemExit) as exc:
        splits.build(Z_BIN, BAND, seed=42, root=tmp_path)
    message = str(exc.value)
    assert "does not describe" in message
    assert "60" in message and "20" in message


@needs_cache
@needs_info
def test_cutout_size_cannot_exceed_the_cache():
    manifest = json.loads(layout.cache_manifest_path(Z_BIN).read_text())
    assert config.target_crop_px(Z_BIN) <= manifest["crop_px"]


@needs_cache
def test_pixel_zp_guard_agrees_with_the_real_manifest():
    manifest = json.loads(layout.cache_manifest_path(Z_BIN).read_text())
    train.check_pixel_zp(Z_BIN, manifest["pixel_zp"])
