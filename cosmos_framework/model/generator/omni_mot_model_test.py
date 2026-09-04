# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace

import torch

from cosmos_framework.model.generator import omni_mot_model
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class _DeviceRecorder:
    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> None:
        self.moves.append(device)


def _model_with_vae(vae: _DeviceRecorder) -> OmniMoTModel:
    model = OmniMoTModel.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.tokenizer_vision_gen = SimpleNamespace(model=SimpleNamespace(model=vae))
    return model


def test_vae_cpu_offload_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(omni_mot_model, "VAE_CPU_OFFLOAD", True)
    empty_cache_calls = 0

    def empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    vae = _DeviceRecorder()

    with _model_with_vae(vae)._vae_on_gpu():
        assert vae.moves == ["cuda"]

    assert vae.moves == ["cuda", "cpu"]
    assert empty_cache_calls == 1


def test_vae_cpu_offload_disabled(monkeypatch) -> None:
    monkeypatch.setattr(omni_mot_model, "VAE_CPU_OFFLOAD", False)
    vae = _DeviceRecorder()

    with _model_with_vae(vae)._vae_on_gpu():
        pass

    assert vae.moves == []
