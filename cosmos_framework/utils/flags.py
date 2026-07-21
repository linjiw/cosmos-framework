# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Feature flags."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Final


class StrEnum(str, Enum):
    """Backport of StrEnum from Python 3.11."""

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.lower()


def _parse_bool(value: str) -> bool:
    """Parse string to a boolean."""
    return value.lower() in ["true", "1", "yes", "y"]


def _get_bool(name: str, default: bool) -> bool:
    """Get a boolean flag from the environment."""
    value = os.environ.get(name, "")
    if not value:
        return default
    return _parse_bool(value)


TRAINING: Final[bool] = _get_bool("COSMOS_TRAINING", True)
"""Whether to enable training features.

This is used to make training dependencies optional.
"""

INTERNAL: Final[bool] = _get_bool("COSMOS_INTERNAL", False)
"""Whether to use internal (nvidia-only) resources (e.g. S3)."""

SMOKE: Final[bool] = _get_bool("COSMOS_SMOKE", False)
"""Whether to enable smoke test.

Sets parameters to minimum values (e.g. num_steps=1, num_layers=2).
"""


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    META = "meta"


DEVICE: Final[Device] = Device(os.environ.get("COSMOS_DEVICE", "cuda").lower())
"""Torch device to use.

Used for checkpoint conversion and smoke tests.
"""

VERBOSE: Final[bool] = _get_bool("COSMOS_VERBOSE", INTERNAL)
"""Whether to enable verbose console output."""

VAE_CPU_OFFLOAD: Final[bool] = _get_bool("COSMOS_VAE_CPU_OFFLOAD", False)
"""Keep the vision VAE tokenizer on CPU and move it to GPU only for encode/decode calls.

Saves the VAE's resident GPU memory (~1.3 GiB bf16 for Wan2.2) on memory-constrained
single-GPU training at the cost of a PCIe round-trip per call. Encode is deterministic,
so results are identical to keeping the VAE resident."""

EMA_CPU_SUBSET: Final[bool] = _get_bool("COSMOS_EMA_CPU_SUBSET", False)
"""Replace the full fp32 GPU ``net_ema`` clone with a CPU-resident EMA over the
trainable-parameter subset only.

During subset fine-tuning (``keys_to_select``) frozen parameters never change, so
their EMA is identically their initial value — tracking only trainable params in
pinned host memory is bit-equivalent to the stock full clone (~12.5 GiB GPU for
Cosmos3-Edge) at zero GPU cost. See utils/generator/cpu_subset_ema.py."""

EXPERIMENTAL_CHECKPOINTS: Final[bool] = _get_bool("COSMOS_EXPERIMENTAL_CHECKPOINTS", INTERNAL)
"""Whether to enable experimental checkpoints."""


if INTERNAL:
    TRAINING = True


@dataclass
class Flags:
    internal: bool = INTERNAL
    training: bool = TRAINING
    smoke: bool = SMOKE
    device: Device = DEVICE
    verbose: bool = VERBOSE
    experimental_checkpoints: bool = EXPERIMENTAL_CHECKPOINTS


FLAGS = Flags()
"""Convenience object for accessing flags."""
