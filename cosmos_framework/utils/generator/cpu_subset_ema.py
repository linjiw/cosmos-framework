# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-resident, trainable-subset EMA for memory-constrained single-GPU SFT.

Rationale (see work/cosmos3edge/OPTIMIZATION_PLAN.md): the stock ``net_ema`` is a
full fp32 GPU clone of the whole tower (~12.5 GiB for Cosmos3-Edge). During subset
fine-tuning (``keys_to_select``) only the selected parameters ever change, so the
EMA of every frozen parameter is identically its initial value. Keeping fp32 EMA
buffers for ONLY the trainable subset, resident in pinned CPU memory and updated
with the exact same lerp (``ema = beta * ema + (1-beta) * param``), produces
bit-equivalent EMA state at zero GPU cost.

Interface mirrors the places OmniMoTModel consumes (net_ema, net_ema_worker):

* ``update_average(net, beta)``   — D2H snapshot of trainable params + host lerp.
* ``copy_to_from(net)``           — (re)seed EMA buffers from the net.
* ``swap_into(net)/swap_back(net)`` — ema_scope support: temporarily load EMA
  values into the net's trainable params (validation/sampling), then restore.
* ``ema_state_dict(net, prefix)`` — full ``net_ema.*``-compatible state dict:
  EMA buffers for trainable keys + fp32 casts of net values for frozen keys
  (mathematically exact, since frozen EMA == frozen weight).
* ``load_ema_state_dict(sd)``     — route incoming net_ema.* values into buffers.
"""

from __future__ import annotations

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor as dt2lt


def _canon(name: str) -> str:
    """Canonicalize a named_parameters() name to state_dict key format.

    Activation-checkpoint wrappers inject ``_checkpoint_wrapped_module`` segments
    into attribute paths; the wrapper's state-dict hooks strip them, so checkpoint
    keys never contain them (torch DCP's FQN validation rejects keys that do).
    ``_orig_mod`` (torch.compile) IS kept — stock state_dict keys retain it and
    DCP's FQN walk handles it.
    """
    return name.replace("._checkpoint_wrapped_module", "")


class CPUSubsetEMA:
    """fp32 EMA over the ``keys_to_select`` parameter subset, pinned on CPU.

    Buffers are keyed by canonical state-dict names (see :func:`_canon`) so the
    reconstructed ``net_ema.*`` checkpoint entries match the stock key format.
    """

    def __init__(self, net: torch.nn.Module, keys_to_select: list[str]):
        assert len(keys_to_select) > 0, "CPUSubsetEMA requires a non-empty keys_to_select"
        self.keys_to_select = list(keys_to_select)
        self._buffers: dict[str, torch.Tensor] = {}
        self._staging: dict[str, torch.Tensor] = {}
        self._swap_cache: list[torch.Tensor] | None = None
        for name, p in net.named_parameters():
            if any(k in name for k in self.keys_to_select):
                cname = _canon(name)
                local = dt2lt(p).detach()
                self._buffers[cname] = local.to(device="cpu", dtype=torch.float32).pin_memory()
                # bf16 staging buffer for async D2H of the live param
                self._staging[cname] = torch.empty_like(local, device="cpu").pin_memory()
        n_params = sum(b.numel() for b in self._buffers.values())
        log.info(
            f"CPUSubsetEMA: tracking {len(self._buffers)} tensors / {n_params / 1e9:.3f}B params "
            f"({n_params * 4 / 1024**3:.2f} GiB pinned host fp32)"
        )

    # -- stock-net_ema_worker replacements ---------------------------------

    @torch.no_grad()
    def copy_to_from(self, net: torch.nn.Module) -> None:
        """Seed EMA buffers from the current net weights (fp32 upcast)."""
        for name, p in net.named_parameters():
            name = _canon(name)
            if name in self._buffers:
                self._buffers[name].copy_(dt2lt(p).detach().to(device="cpu", dtype=torch.float32))

    @torch.no_grad()
    def update_average(self, net: torch.nn.Module, beta: float) -> None:
        """ema = beta * ema + (1 - beta) * param — same math as DTensorFastEmaModelUpdater."""
        names = []
        for name, p in net.named_parameters():
            name = _canon(name)
            if name in self._buffers:
                self._staging[name].copy_(dt2lt(p).detach(), non_blocking=True)
                names.append(name)
        torch.cuda.synchronize()  # staging buffers must be complete before host math
        targets = [self._buffers[n] for n in names]
        sources = [self._staging[n].to(torch.float32) for n in names]
        torch._foreach_mul_(targets, beta)
        torch._foreach_add_(targets, sources, alpha=1.0 - beta)

    # -- ema_scope support ---------------------------------------------------

    @torch.no_grad()
    def swap_into(self, net: torch.nn.Module) -> None:
        """Cache live trainable params (CPU) and load EMA values into the net."""
        assert self._swap_cache is None, "CPUSubsetEMA swap already active"
        cache = []
        for name, p in net.named_parameters():
            name = _canon(name)
            if name in self._buffers:
                local = dt2lt(p)
                cache.append(local.detach().to("cpu", copy=True))
                local.data.copy_(self._buffers[name].to(device=local.device, dtype=local.dtype))
        self._swap_cache = cache

    @torch.no_grad()
    def swap_back(self, net: torch.nn.Module) -> None:
        """Restore the live params cached by :meth:`swap_into`."""
        assert self._swap_cache is not None, "CPUSubsetEMA swap not active"
        it = iter(self._swap_cache)
        for name, p in net.named_parameters():
            name = _canon(name)
            if name in self._buffers:
                local = dt2lt(p)
                local.data.copy_(next(it).to(device=local.device, dtype=local.dtype))
        self._swap_cache = None

    # -- checkpoint integration ----------------------------------------------

    @torch.no_grad()
    def ema_state_dict(self, net: torch.nn.Module, prefix: str = "") -> dict[str, torch.Tensor]:
        """Full net_ema-compatible state dict.

        Trainable keys -> EMA buffers (fp32). Frozen keys -> fp32 casts of the
        live net values, which equal the stock net_ema exactly (frozen params
        never change, and the stock worker's copy_to seeded them identically).
        """
        out: dict[str, torch.Tensor] = {}
        for name, p in net.named_parameters():
            name = _canon(name)
            if name in self._buffers:
                out[prefix + name] = self._buffers[name].clone()
            else:
                out[prefix + name] = dt2lt(p).detach().to(device="cpu", dtype=torch.float32)
        # buffers (non-parameter state, e.g. positional caches) mirror the net,
        # matching the stock worker which only lerps parameters.
        for name, b in net.named_buffers():
            out[prefix + _canon(name)] = dt2lt(b).detach().to(device="cpu")
        return out

    @torch.no_grad()
    def load_ema_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load net_ema.* values (already stripped of the prefix) into EMA buffers."""
        hit = 0
        for name, v in state_dict.items():
            name = _canon(name)
            if name in self._buffers:
                self._buffers[name].copy_(v.to(device="cpu", dtype=torch.float32))
                hit += 1
        log.info(f"CPUSubsetEMA: loaded {hit}/{len(self._buffers)} EMA tensors from checkpoint")
