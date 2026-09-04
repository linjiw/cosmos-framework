# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Low-precision quantization helpers for the Cosmos3 VFM MoT network.

Quantization is applied via torchao's :func:`apply_quantization_inplace`, which
uses the ``quantize_`` path to replace each selected weight with a quantized
tensor subclass in place. Because the live parameter becomes a tensor subclass,
this only works on unsharded (plain-tensor) params and is therefore restricted
to replicated inference (``data_parallel_shard_degree == 1``); it cannot be
applied to an FSDP-sharded model whose params are ``DTensor`` shards.

This is an inference-only path: the ``quantize_`` PTQ configs have no backward
support. Module selection is delegated to the filter built by
:func:`_get_filter_fn`.
"""

import re
from typing import Any

import torch
from torch import nn

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

# NOTE: ``torchao`` is imported lazily inside the functions below rather than at
# module top level. These two helpers are the only torchao consumers, but this
# module is imported transitively by the model package (e.g. during tests that
# never quantize). Keeping the imports lazy means importing this module does not
# require torchao to be installed; the imports only run — and only fail — when
# quantization is actually requested.


def _get_filter_fn(quantization_config: QuantizationConfig):
    """Build a module-selection predicate from the quantization config.

    The returned closure captures ``include_regex`` / ``exclude_regex`` and
    implements the selection policy documented on :class:`QuantizationConfig`.
    Each key is treated as a regular expression and matched against a module's
    FQN with :func:`re.search` (a plain substring remains a valid pattern, so
    existing substring-style keys keep working). It is passed to torchao as
    ``filter_fn`` (for ``quantize_``), which expects a
    ``(module, fqn) -> bool`` signature.

    Args:
        quantization_config: Config carrying the include/exclude key lists.

    Returns:
        A predicate suitable for torchao's ``filter_fn`` / ``module_filter_fn``.
    """

    def _filter_fn(mod: nn.Module, name: str) -> bool:
        """Decide whether a single module should be quantized.

        Called once per module as torchao walks the model recursively. A module
        is selected only when ALL of the following hold:

        1. It is an ``nn.Linear`` (the only layer type these recipes support).
        2. ``include_regex`` is empty (include everything) OR the module's FQN
           matches at least one include pattern.
        3. The module's FQN matches none of the ``exclude_regex`` patterns.

        Each include/exclude key is treated as a regular expression and matched
        against the FQN with :func:`re.search`, so the pattern can match anywhere
        in the name (a plain substring is still a valid regex, preserving the
        previous substring-match behavior, while enabling anchors like ``^``/``$``,
        alternation, character classes, etc.).

        Note the parenthesization around the include check: ``and`` binds tighter
        than ``or`` in Python, so without it the ``nn.Linear`` and exclude
        checks would not apply across both include branches.

        Args:
            mod (torch.nn.Module): The module that is being processed.
            name (str): A fully qualified name of the module that is being processed.

        Return:
            True if the module should be quantized, False otherwise.
        """
        include_keys = quantization_config.include_regex
        exclude_keys = quantization_config.exclude_regex
        return (
            isinstance(mod, nn.Linear)
            and (len(include_keys) == 0 or any(re.search(key, name) for key in include_keys))
            and not any(re.search(key, name) for key in exclude_keys)
        )

    return _filter_fn


def should_quantize_module(module: nn.Module, fqn: str, quantization_config: QuantizationConfig) -> bool:
    """Return whether ``module`` is selected by a quantization recipe."""
    return _get_filter_fn(quantization_config)(module, fqn)


def _gate_cuda_woq_pattern_to_cpu():
    """Restore torch<=2.9 behavior: weight-only-quant pattern rewrite on CPU only.

    torch 2.10 enabled an inductor pattern (pytorch#161680) that rewrites the
    int8 dequant+mm graph into ``aten._weight_int8pack_mm`` on CUDA. That CUDA
    kernel is a thread-per-output GEMV — measured ~150x slower than the
    dequant+cublas route it replaces at diffusion batch sizes (M~12k) on
    sm_86. Gating the pattern back to CPU keeps compiled int8wo at ~bf16
    speed. Must run before the first compiled forward of a quantized module
    (the pattern's validity check is captured at lazy registration).
    """
    try:
        from torch._inductor.fx_passes import quantization as _q

        orig = _q._is_valid_woq_optimization_pattern

        def _cpu_only_check(*args, **kwargs):
            inner = orig(*args, **kwargs)

            def check(match):
                if not inner(match):
                    return False
                x = match.kwargs.get("x")
                return x is not None and x.meta["val"].device.type == "cpu"

            return check

        # Idempotent: only patch once.
        if getattr(_q._is_valid_woq_optimization_pattern, "_cosmos_cpu_gated", False):
            return
        _cpu_only_check._cosmos_cpu_gated = True  # type: ignore[attr-defined]
        _q._is_valid_woq_optimization_pattern = _cpu_only_check
    except Exception:
        # Older/newer torch without the pattern: nothing to gate.
        pass


def _get_torchao_config(method: str) -> Any:
    """Construct the torchao recipe shared by bulk and streaming PTQ."""
    if method == "int8wo":
        from torchao.quantization import Int8WeightOnlyConfig

        _gate_cuda_woq_pattern_to_cpu()
        return Int8WeightOnlyConfig(version=2, set_inductor_config=False)
    if method == "int8dq":
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig

        return Int8DynamicActivationInt8WeightConfig(version=2, set_inductor_config=False)
    if method == "mxfp8":
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

        return MXDynamicActivationMXWeightConfig()
    if method == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        # Avoid the external mslk ABI dependency in NGC torch builds.
        return NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    raise ValueError(f"Unsupported quantization method: {method}")


def quantize_linear_weight(weight: torch.Tensor, method: str) -> torch.Tensor:
    """Quantize one CPU Linear weight without first materializing BF16 on CUDA.

    This deliberately uses torchao's public ``quantize_`` transformation on a
    temporary one-layer module instead of constructing its tensor subclass by
    hand. The resulting representation is therefore exactly the same as the
    normal post-load quantization path, including dynamic-activation metadata.
    """
    if method not in {"int8wo", "int8dq"}:
        raise ValueError(f"Streaming quantization does not support method {method!r}.")
    if weight.ndim != 2:
        raise ValueError(f"Linear weights must be two-dimensional, got shape {tuple(weight.shape)}.")
    if weight.device.type != "cpu":
        weight = weight.cpu()

    from torchao.quantization import quantize_

    linear = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        device="meta",
        dtype=weight.dtype,
    )
    linear.weight = nn.Parameter(weight, requires_grad=False)
    quantize_(linear, config=_get_torchao_config(method))
    return linear.weight


def apply_quantization_inplace(model: nn.Module, quantization_config: QuantizationConfig):
    """Apply quantization in place via ``quantize_`` (replaces weights with quantized tensors).

    This is the replication path. ``quantize_`` replaces each weight with a
    quantized tensor subclass as the live parameter, which only works when the
    parameters are plain tensors. It therefore cannot be applied to an already
    FSDP-sharded model (the params are ``DTensor`` shards), so it is restricted
    to replicated inference (``data_parallel_shard_degree == 1``).

    These configs (``MXDynamicActivationMXWeightConfig`` /
    ``NVFP4DynamicActivationNVFP4WeightConfig`` — Blackwell tensor cores — and
    ``Int8WeightOnlyConfig`` — Ampere+) are inference-only (PTQ) and have no
    backward support. For the sharded case use ``apply_quantization``
    (the module-swap path) instead; both functions are currently inference
    paths, selected by whether FSDP is sharding the model.
    """
    # No-op when quantization is disabled.
    if quantization_config.method is None:
        return

    from torchao.quantization import quantize_

    if quantization_config.method == "int8wo":
        # int8 weight-only: weights stored int8 (per-channel scales), compute
        # stays bf16 (dequant per matmul). No fp8/fp4 tensor cores needed, so
        # this is the method that runs on pre-Blackwell GPUs (Ampere+). Unlike
        # the dynamic-activation configs above it quantizes weights only.
        # set_inductor_config=False: the default would flip global inductor
        # flags, changing how the model's non-quantized remainder compiles and
        # confounding any quantized-vs-bf16 comparison.
        quantize_(
            model,
            config=_get_torchao_config(quantization_config.method),
            filter_fn=_get_filter_fn(quantization_config),
        )
    elif quantization_config.method == "int8dq":
        # int8 dynamic-activation + int8 weight (W8A8): routes matmuls through
        # torch._int_mm (INT8 tensor cores, sm_75+). Same int8 weight storage
        # as int8wo (same memory saving) but compute-bound layers get FASTER
        # than bf16 at diffusion batch sizes. Requires torch.compile for speed
        # (eager W8A8 is very slow); adds per-token activation rounding on top
        # of weight rounding — revalidate quality vs int8wo before adopting.
        quantize_(
            model,
            config=_get_torchao_config(quantization_config.method),
            filter_fn=_get_filter_fn(quantization_config),
        )
    elif quantization_config.method == "mxfp8":
        # mxfp8 / nvfp4 use fixed block scales.
        quantize_(
            model,
            config=_get_torchao_config(quantization_config.method),
            filter_fn=_get_filter_fn(quantization_config),
        )
    elif quantization_config.method == "nvfp4":
        # use_triton_kernel=False avoids torchao's fused NVFP4 Triton kernel, which
        # requires the external `mslk` package. Prebuilt mslk wheels are linked
        # against upstream torch and fail to load against NVIDIA's NGC custom torch
        # builds (ABI mismatch on `torch::Library::_def`), so we use torchao's
        # built-in NVFP4 path instead.
        quantize_(
            model,
            config=_get_torchao_config(quantization_config.method),
            filter_fn=_get_filter_fn(quantization_config),
        )
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_config.method}")
