import pytest
import torch

from cogeneration.models.esm_ckpt_loading import plan_esm_warm_start_state_dict_load


def test_esm_warm_start_allows_missing_pair_keys_when_requested():
    current = {
        "model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4),
        "model.esm_combiner.pair_proj.0.weight": torch.zeros(8, 8),
        "model.esm_combiner.pair_layer_norm.weight": torch.zeros(8),
        "model.trunk.some.weight": torch.zeros(1),
    }
    ckpt = {
        "model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4),
        "model.trunk.some.weight": torch.zeros(1),
    }

    plan = plan_esm_warm_start_state_dict_load(
        checkpoint_state_dict=ckpt,
        current_state_dict=current,
        strict=True,
        allow_missing_esm_combiner_pair=True,
    )

    assert plan.strict is False
    assert any(".esm_combiner.pair_proj." in k for k in plan.ignored_missing_keys)


def test_esm_warm_start_rejects_missing_pair_keys_by_default():
    current = {
        "model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4),
        "model.esm_combiner.pair_proj.0.weight": torch.zeros(8, 8),
    }
    ckpt = {
        "model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4),
    }

    with pytest.raises(RuntimeError, match="Missing keys"):
        plan_esm_warm_start_state_dict_load(
            checkpoint_state_dict=ckpt,
            current_state_dict=current,
            strict=True,
            allow_missing_esm_combiner_pair=False,
        )


def test_esm_warm_start_rejects_esm_combiner_if_disabled():
    current = {"model.trunk.some.weight": torch.zeros(1)}
    ckpt = {"model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4)}

    with pytest.raises(RuntimeError, match="ESM combiner keys found in checkpoint"):
        plan_esm_warm_start_state_dict_load(
            checkpoint_state_dict=ckpt,
            current_state_dict=current,
            strict=True,
            allow_missing_esm_combiner_pair=False,
        )


def test_esm_warm_start_rejects_esm_combiner_size_mismatch():
    current = {"model.esm_combiner.seq_proj.0.weight": torch.zeros(4, 4)}
    ckpt = {"model.esm_combiner.seq_proj.0.weight": torch.zeros(5, 4)}

    with pytest.raises(RuntimeError, match="ESM combiner size mismatch"):
        plan_esm_warm_start_state_dict_load(
            checkpoint_state_dict=ckpt,
            current_state_dict=current,
            strict=True,
            allow_missing_esm_combiner_pair=False,
        )
