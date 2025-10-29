import math
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn


ExpertSelector = Optional[Union[int, Sequence[int], Iterable[int], torch.Tensor]]


class LinearLoRASquared(nn.Linear):
    """
    Linear layer augmented with one shared LoRA branch and a pool of expert LoRA branches.

    The shared branch is always active, while expert branches contribute only when an
    index (or collection of indices) is passed to the forward method.

    Args:
        existing_linear: The pretrained linear layer to augment.
        r_shared: Rank of the shared LoRA adapter. Set to 0 to disable.
        r_expert: Rank of each expert adapter. Set to 0 to disable experts.
        n_experts: Number of expert adapters to instantiate.
        alpha_shared: Scaling factor for shared LoRA (alpha / r convention).
        alpha_expert: Scaling factor for expert LoRA branches.
        dropout_rate: Dropout applied to the input before the low-rank projections.
        fan_in_fan_out: Flag mirroring the LoRA convention for weight orientation.
        freeze_base: If True, keeps the original weight/bias frozen.
    """

    def __init__(
        self,
        existing_linear: nn.Linear,
        r_shared: int,
        r_expert: int,
        n_experts: int,
        alpha_shared: float = 1.0,
        alpha_expert: float = 1.0,
        dropout_rate: float = 0.0,
        fan_in_fan_out: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
            bias=existing_linear.bias is not None,
        )

        self.load_state_dict(existing_linear.state_dict())

        self.r_shared = r_shared
        self.r_expert = r_expert
        self.n_experts = n_experts
        self.alpha_shared = alpha_shared
        self.alpha_expert = alpha_expert
        self.fan_in_fan_out = fan_in_fan_out

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.t()

        self.scaling_shared = alpha_shared / r_shared if r_shared > 0 else 0.0
        self.scaling_expert = alpha_expert / r_expert if r_expert > 0 else 0.0
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        # Shared LoRA branch
        if self.r_shared > 0:
            self.lora_shared_A = nn.Parameter(
                torch.zeros(self.r_shared, self.in_features)
            )
            self.lora_shared_B = nn.Parameter(
                torch.zeros(self.out_features, self.r_shared)
            )
            self._reset_shared_parameters()
        else:
            self.register_parameter("lora_shared_A", None)
            self.register_parameter("lora_shared_B", None)

        # Expert LoRA branches
        if self.r_expert > 0 and self.n_experts > 0:
            self.lora_expert_A = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(self.r_expert, self.in_features))
                    for _ in range(self.n_experts)
                ]
            )
            self.lora_expert_B = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(self.out_features, self.r_expert))
                    for _ in range(self.n_experts)
                ]
            )
            self._reset_expert_parameters()
        else:
            self.lora_expert_A = nn.ParameterList()
            self.lora_expert_B = nn.ParameterList()

    def _reset_shared_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_shared_B)

    def _reset_expert_parameters(self) -> None:
        for param in self.lora_expert_A:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in self.lora_expert_B:
            nn.init.zeros_(param)

    def _drop_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None and self.training:
            return self.dropout(x)
        return x

    def _normalize_indices(self, expert_index: ExpertSelector) -> List[int]:
        if expert_index is None:
            return []

        if isinstance(expert_index, int):
            indices = [expert_index]
        elif isinstance(expert_index, torch.Tensor):
            if expert_index.numel() == 1:
                indices = [int(expert_index.item())]
            else:
                raise ValueError(
                    "Per-sample expert selection is not supported; provide a scalar index."
                )
        elif isinstance(expert_index, (list, tuple, set)):
            indices = list(expert_index)
        else:
            indices = list(expert_index)

        for idx in indices:
            if not 0 <= idx < self.n_experts:
                raise IndexError(
                    f"Expert index {idx} is out of range for {self.n_experts} experts."
                )
        return indices

    def _apply_shared(self, dropped: torch.Tensor) -> torch.Tensor:
        if self.r_shared == 0:
            return torch.zeros(
                dropped.shape[0], self.out_features, dtype=dropped.dtype, device=dropped.device
            )
        update = dropped @ self.lora_shared_A.t()
        update = update @ self.lora_shared_B.t()
        return update * self.scaling_shared

    def _apply_experts(
        self, dropped: torch.Tensor, indices: Sequence[int]
    ) -> torch.Tensor:
        if self.r_expert == 0 or len(indices) == 0:
            return torch.zeros(
                dropped.shape[0], self.out_features, dtype=dropped.dtype, device=dropped.device
            )
        update = None
        for idx in indices:
            proj = dropped @ self.lora_expert_A[idx].t()
            proj = proj @ self.lora_expert_B[idx].t()
            proj = proj * self.scaling_expert
            update = proj if update is None else update + proj
        if update is None:
            return torch.zeros(
                dropped.shape[0], self.out_features, dtype=dropped.dtype, device=dropped.device
            )
        return update

    def forward(
        self, x: torch.Tensor, expert_index: ExpertSelector = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_features).
            expert_index: Optional expert id or iterable of ids to activate. If None,
                only the shared LoRA branch is used.
        """
        result = nn.functional.linear(x, self.weight, self.bias)
        dropped = self._drop_input(x)

        if self.r_shared > 0:
            result = result + self._apply_shared(dropped)

        indices = self._normalize_indices(expert_index)
        if indices:
            result = result + self._apply_experts(dropped, indices)

        return result

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, r_shared={self.r_shared}, r_expert={self.r_expert}, "
            f"n_experts={self.n_experts}, alpha_shared={self.alpha_shared}, "
            f"alpha_expert={self.alpha_expert}"
        )
