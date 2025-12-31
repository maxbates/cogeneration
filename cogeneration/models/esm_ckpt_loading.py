"""
Utility for loading model checkpoints that omit ESM weights
(since they are frozen and loaded separately)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch

_FROZEN_ESM_KEY_SUBSTRING = ".esm_combiner.esm."
_ESM_COMBINER_KEY_SUBSTRING = "esm_combiner"
_ESM_COMBINER_PAIR_KEY_SUBSTRINGS = (
    ".esm_combiner.pair_proj.",
    ".esm_combiner.pair_layer_norm.",
)


@dataclass(frozen=True)
class StateDictLoadPlan:
    state_dict: Dict[str, torch.Tensor]
    strict: bool
    ignored_missing_keys: Tuple[str, ...] = ()
    ignored_unexpected_keys: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()


def _matches_any_substring(key: str, substrings: Sequence[str]) -> bool:
    return any(s in key for s in substrings)


def plan_state_dict_load(
    *,
    checkpoint_state_dict: Mapping[str, torch.Tensor],
    current_state_dict: Mapping[str, torch.Tensor],
    strict: bool,
    allow_missing_key_substrings: Sequence[str] = (),
    allow_unexpected_key_substrings: Sequence[str] = (),
) -> StateDictLoadPlan:
    """
    Plan a safe `load_state_dict` call by optionally relaxing `strict` while still
    validating that any missing/unexpected keys are explicitly allowed.

    This mirrors the common warm-start pattern:
    - Keep `strict=True` unless there are *known-safe* missing/unexpected keys.
    - If there are disallowed missing/unexpected keys, raise an error early.
    """
    if not strict:
        return StateDictLoadPlan(state_dict=dict(checkpoint_state_dict), strict=False)

    checkpoint_keys = set(checkpoint_state_dict.keys())
    current_keys = set(current_state_dict.keys())

    missing_keys = sorted(current_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - current_keys)

    ignored_missing_keys = tuple(
        k
        for k in missing_keys
        if _matches_any_substring(k, allow_missing_key_substrings)
    )
    ignored_unexpected_keys = tuple(
        k
        for k in unexpected_keys
        if _matches_any_substring(k, allow_unexpected_key_substrings)
    )

    disallowed_missing_keys = [k for k in missing_keys if k not in ignored_missing_keys]
    disallowed_unexpected_keys = [
        k for k in unexpected_keys if k not in ignored_unexpected_keys
    ]

    if disallowed_missing_keys or disallowed_unexpected_keys:
        msg = ["Cannot load checkpoint with strict=True due to incompatible keys."]
        if disallowed_missing_keys:
            msg.append(
                f"Missing keys ({len(disallowed_missing_keys)}): {disallowed_missing_keys}"
            )
        if disallowed_unexpected_keys:
            msg.append(
                f"Unexpected keys ({len(disallowed_unexpected_keys)}): {disallowed_unexpected_keys}"
            )
        raise RuntimeError("\n".join(msg))

    # If we got here, any differences are explicitly allowed: relax strict.
    relaxed = bool(ignored_missing_keys or ignored_unexpected_keys)
    return StateDictLoadPlan(
        state_dict=dict(checkpoint_state_dict),
        strict=(False if relaxed else True),
        ignored_missing_keys=ignored_missing_keys,
        ignored_unexpected_keys=ignored_unexpected_keys,
    )


def plan_esm_warm_start_state_dict_load(
    *,
    checkpoint_state_dict: Mapping[str, torch.Tensor],
    current_state_dict: Mapping[str, torch.Tensor],
    strict: bool,
    allow_missing_esm_combiner_pair: bool = False,
) -> StateDictLoadPlan:
    """
    Warm-start compatibility for models that use `ESMCombinerNetwork` with a frozen
    ESM backbone.

    - The frozen ESM weights (keys containing `.esm_combiner.esm.`) may be omitted
      from checkpoints because they are loaded separately.
    - If `only_single` differs between checkpoint/current config, the ESM combiner
      pair projection keys may be absent/present and can optionally be ignored.
    - Still validates that the rest of the checkpoint matches the current model.
    """
    if not strict:
        return StateDictLoadPlan(state_dict=dict(checkpoint_state_dict), strict=False)

    checkpoint_keys = set(checkpoint_state_dict.keys())
    current_keys = set(current_state_dict.keys())

    esm_enabled = any(_ESM_COMBINER_KEY_SUBSTRING in k for k in current_keys)
    checkpoint_has_esm_combiner = any(
        _ESM_COMBINER_KEY_SUBSTRING in k for k in checkpoint_keys
    )
    checkpoint_has_frozen_esm = any(
        _FROZEN_ESM_KEY_SUBSTRING in k for k in checkpoint_keys
    )

    if not esm_enabled and checkpoint_has_esm_combiner:
        raise RuntimeError(
            "Cannot warm start: ESM combiner keys found in checkpoint but ESM is not enabled."
        )

    # Give a clearer error for ESM combiner dimension mismatch (wrong ESM model size).
    if esm_enabled:
        mismatches = []
        for key in checkpoint_keys & current_keys:
            if (
                _ESM_COMBINER_KEY_SUBSTRING in key
                and _FROZEN_ESM_KEY_SUBSTRING not in key
            ):
                if checkpoint_state_dict[key].shape != current_state_dict[key].shape:
                    mismatches.append(
                        f"{key}: ckpt{tuple(checkpoint_state_dict[key].shape)} vs current{tuple(current_state_dict[key].shape)}"
                    )
        if mismatches:
            raise RuntimeError(
                "Cannot load checkpoint: ESM combiner size mismatch detected.\n"
                "The checkpoint was trained with a different ESM model size.\n"
                f"Mismatched parameters ({len(mismatches)}):\n"
                + "\n".join(f"  - {m}" for m in mismatches[:5])
                + (
                    f"\n  ... and {len(mismatches) - 5} more"
                    if len(mismatches) > 5
                    else ""
                )
                + "\n\nTo fix: Use the same ESM model as the checkpoint, or train from scratch."
            )

    allow_missing_key_substrings = [_FROZEN_ESM_KEY_SUBSTRING]
    allow_unexpected_key_substrings = [_FROZEN_ESM_KEY_SUBSTRING]
    notes = []

    if allow_missing_esm_combiner_pair:
        current_has_pair = any(
            _matches_any_substring(k, _ESM_COMBINER_PAIR_KEY_SUBSTRINGS)
            for k in current_keys
        )
        checkpoint_has_pair = any(
            _matches_any_substring(k, _ESM_COMBINER_PAIR_KEY_SUBSTRINGS)
            for k in checkpoint_keys
        )
        if current_has_pair and not checkpoint_has_pair:
            allow_missing_key_substrings.extend(_ESM_COMBINER_PAIR_KEY_SUBSTRINGS)
            notes.append(
                "Checkpoint has no ESM combiner pair projection weights; loading with strict=False"
            )
        if checkpoint_has_pair and not current_has_pair:
            allow_unexpected_key_substrings.extend(_ESM_COMBINER_PAIR_KEY_SUBSTRINGS)
            notes.append(
                "Checkpoint has ESM combiner pair projection weights but current model does not; loading with strict=False"
            )

    plan = plan_state_dict_load(
        checkpoint_state_dict=checkpoint_state_dict,
        current_state_dict=current_state_dict,
        strict=True,
        allow_missing_key_substrings=allow_missing_key_substrings,
        allow_unexpected_key_substrings=allow_unexpected_key_substrings,
    )

    # if frozen ESM weights are omitted (expected), relax strict.
    if esm_enabled and not checkpoint_has_frozen_esm and plan.strict:
        plan = StateDictLoadPlan(
            state_dict=plan.state_dict,
            strict=False,
            ignored_missing_keys=plan.ignored_missing_keys,
            ignored_unexpected_keys=plan.ignored_unexpected_keys,
            notes=tuple(
                [
                    "Checkpoint has no frozen ESM weights (expected); loading with strict=False",
                    *notes,
                ]
            ),
        )
    elif notes:
        plan = StateDictLoadPlan(
            state_dict=plan.state_dict,
            strict=plan.strict,
            ignored_missing_keys=plan.ignored_missing_keys,
            ignored_unexpected_keys=plan.ignored_unexpected_keys,
            notes=tuple(notes),
        )

    return plan
