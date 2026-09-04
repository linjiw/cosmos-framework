# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig


@pytest.mark.parametrize("method", [None, "mxfp8", "nvfp4", "int8wo", "int8dq"])
def test_quantization_config_accepts_supported_methods(method: str | None) -> None:
    assert QuantizationConfig(method=method).method == method


def test_quantization_config_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="must be in"):
        QuantizationConfig(method="unknown")
