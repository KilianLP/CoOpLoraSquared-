"""
LoRA-Squared library: shared + expert LoRA adapters.

This package complements the original `loralib` utilities by providing
modules that combine a shared LoRA branch with a bank of expert-specific
branches that can be activated dynamically at inference time.
"""

from .layers import LinearLoRASquared
from .utils import (
    PlainMultiheadAttentionLoRASquared,
    apply_lorasquared,
    mark_only_lorasquared_as_trainable,
    get_lorasquared_parameters,
    resolve_expert_indices,
)

__all__ = [
    "LinearLoRASquared",
    "PlainMultiheadAttentionLoRASquared",
    "apply_lorasquared",
    "mark_only_lorasquared_as_trainable",
    "get_lorasquared_parameters",
    "resolve_expert_indices",
]
