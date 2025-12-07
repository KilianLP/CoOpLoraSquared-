import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import clip


def attach_expert_metadata(
    dataset,
    mode: str = "per_class",
    num_experts: Optional[int] = None,
    seed: int = 1,
) -> None:
    """
    Build expert-id mappings across dataset splits and attach lookup tensors.

    Modes:
        - per_class (default): one expert per unique class name (previous behavior).
        - random_balanced: distribute base classes uniformly at random across
          ``num_experts`` experts (difference of at most one class between experts).
    """
    if mode not in {"per_class", "random_balanced"}:
        raise ValueError(f"Unsupported expert assignment mode: {mode}")

    base_classes = list(getattr(dataset, "classnames", []) or [])

    if mode == "random_balanced":
        if num_experts is None or num_experts <= 0:
            raise ValueError("random_balanced mode requires num_experts > 0")
        rng = random.Random(seed)
        shuffled = base_classes.copy()
        rng.shuffle(shuffled)
        per_expert = len(shuffled) // num_experts
        remainder = len(shuffled) % num_experts
        classname_to_expert: Dict[str, int] = {}
        idx = 0
        for expert_id in range(num_experts):
            quota = per_expert + (1 if expert_id < remainder else 0)
            for _ in range(quota):
                if idx >= len(shuffled):
                    break
                classname_to_expert[shuffled[idx]] = expert_id
                idx += 1

        def assign_missing(names):
            if names is None:
                return
            for name in names:
                if name not in classname_to_expert:
                    classname_to_expert[name] = rng.randrange(num_experts)

        assign_missing(getattr(dataset, "val_classnames", None))
        assign_missing(getattr(dataset, "test_classnames", None))
        assign_missing(getattr(dataset, "test_new_classnames", None))

        dataset.num_experts = num_experts
        dataset.classname_to_expert = classname_to_expert
        dataset.expert_classnames = list(classname_to_expert.keys())

    else:
        seen = {}
        ordered = []

        def register(names):
            if names is None:
                return
            for name in names:
                if name not in seen:
                    seen[name] = len(ordered)
                    ordered.append(name)

        register(getattr(dataset, "classnames", None))
        register(getattr(dataset, "val_classnames", None))
        register(getattr(dataset, "test_classnames", None))
        register(getattr(dataset, "test_new_classnames", None))

        dataset.classname_to_expert = seen
        dataset.expert_classnames = ordered
        dataset.num_experts = len(ordered)

    def build_tensor(names):
        if names is None:
            return None
        return torch.tensor(
            [dataset.classname_to_expert[name] for name in names], dtype=torch.long
        )

    dataset.label_to_expert_train = build_tensor(getattr(dataset, "classnames", None))
    dataset.label_to_expert_val = build_tensor(getattr(dataset, "val_classnames", None))
    dataset.label_to_expert_test = build_tensor(getattr(dataset, "test_classnames", None))
    dataset.label_to_expert_test_new = build_tensor(getattr(dataset, "test_new_classnames", None))


def named_modules_with_index(clip_model: nn.Module):
    assert hasattr(clip_model, "visual") and hasattr(clip_model.visual, "transformer") and hasattr(clip_model, "transformer"), \
        "The model should have both vision and text transformer modules! RN not supported, implement it yourself :)"
    total_vision_blocks = len(clip_model.visual.transformer.resblocks)
    total_text_blocks = len(clip_model.transformer.resblocks)
    for name, module in clip_model.named_modules():
        if "ln_post" in name:
            yield name, module, total_vision_blocks
        if "ln_final" in name:
            yield name, module, total_text_blocks 
        if "ln_pre" in name:
            yield name, module, 0
        splitname = name.split('resblocks.')
        if len(splitname) == 1: # not a resblock
            yield name, module, -1
        else:
            block_idx = int(splitname[-1].split('.')[0])
            yield name, module, block_idx


def trainable_norm_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []
    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if isinstance(module, torch.nn.LayerNorm) and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            trainable_params.extend(list(module.parameters()))
            module.requires_grad_(True)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> LayerNorm at {name} is trainable.")
        else:
            module.requires_grad_(False)
    return trainable_params


def trainable_bias_params(model, modality='both', vision_start=0, text_start=0):
    assert modality in ('both', 'vision', 'text')
    trainable_params = []

    for param in model.parameters():
        param.requires_grad_(False)

    for name, module, block_idx in named_modules_with_index(model):
        curr_modality = 'vision' if 'visual' in name else 'text'
        curr_index = vision_start if curr_modality == 'vision' else text_start
        if hasattr(module, "bias") and block_idx >= curr_index and (modality == 'both' or modality == curr_modality):
            module.bias.requires_grad_(True)
            trainable_params.append(module.bias)
            print(f"Modality = {modality}, vision_start={vision_start}, text_start={text_start} ==> Bias at {name}.bias is trainable.")
    
    return trainable_params


def num_params(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
