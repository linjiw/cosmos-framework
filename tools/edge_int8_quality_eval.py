#!/usr/bin/env python3
"""Compare paired BF16, INT8WO, and INT8DQ Cosmos3-Edge outputs.

Expected layout beneath ``--root``::

    t2i_bf16/t2i/0/vision.jpg
    t2i_int8wo/t2i/0/vision.jpg
    t2i_int8dq/t2i/0/vision.jpg
    i2v_bf16/i2v/0/vision.mp4
    ...

The numbered directories are produced by the inference CLI when
``--num-outputs`` is greater than one.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import lpips
import numpy as np
import open_clip
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MODES = ("bf16", "int8wo", "int8dq")
PAIR_NAMES = (("int8wo", "bf16"), ("int8dq", "bf16"), ("int8dq", "int8wo"))


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def _lpips_distance(model: lpips.LPIPS, left: np.ndarray, right: np.ndarray) -> float:
    left_tensor = torch.from_numpy(left).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)
    right_tensor = torch.from_numpy(right).permute(2, 0, 1).unsqueeze(0).mul(2).sub(1)
    with torch.inference_mode():
        return float(model(left_tensor, right_tensor).item())


def _clip_image_embeddings(
    model: torch.nn.Module,
    preprocess: Any,
    images: Sequence[np.ndarray],
) -> torch.Tensor:
    batch = torch.stack([preprocess(Image.fromarray((image * 255).round().astype(np.uint8))) for image in images])
    with torch.inference_mode():
        features = model.encode_image(batch)
    return features / features.norm(dim=-1, keepdim=True)


def _clip_prompt_score(
    model: torch.nn.Module,
    preprocess: Any,
    tokenizer: Any,
    images: Sequence[np.ndarray],
    prompt: str,
) -> float:
    image_features = _clip_image_embeddings(model, preprocess, images)
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer([prompt]))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return float((image_features @ text_features.T).mean().item())


def _clip_image_similarity(
    model: torch.nn.Module,
    preprocess: Any,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    features = _clip_image_embeddings(model, preprocess, [left, right])
    return float((features[0] @ features[1]).item())


def _pair_metrics(model: lpips.LPIPS, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "psnr": float(peak_signal_noise_ratio(right, left, data_range=1.0)),
        "ssim": float(structural_similarity(right, left, data_range=1.0, channel_axis=2)),
        "lpips": _lpips_distance(model, left, right),
    }


def _mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _temporal_metrics(model: lpips.LPIPS, frames: Sequence[np.ndarray]) -> dict[str, float]:
    adjacent = [_pair_metrics(model, frames[i], frames[i - 1]) for i in range(1, len(frames))]
    flow_magnitudes: list[float] = []
    for index in range(1, len(frames)):
        previous = cv2.cvtColor((frames[index - 1] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        current = cv2.cvtColor((frames[index] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitudes.append(float(np.linalg.norm(flow, axis=2).mean()))
    return {
        "adjacent_ssim": float(np.mean([row["ssim"] for row in adjacent])),
        "adjacent_lpips": float(np.mean([row["lpips"] for row in adjacent])),
        "flow_magnitude": float(np.mean(flow_magnitudes)),
        "flow_roughness": float(np.mean(np.abs(np.diff(flow_magnitudes)))) if len(flow_magnitudes) > 1 else 0.0,
    }


def _aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    first = next(iter(per_seed.values()))
    for section, section_value in first.items():
        if not isinstance(section_value, dict):
            continue
        aggregate[section] = {}
        for key in section_value:
            values = [float(seed_value[section][key]) for seed_value in per_seed.values()]
            aggregate[section][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    return aggregate


def _promotion_gate(aggregate: dict[str, Any], modalities: Sequence[str]) -> dict[str, Any]:
    """Apply conservative, pre-registered quality-retention thresholds."""
    checks: dict[str, bool] = {}
    for modality in modalities:
        metrics = aggregate[modality]
        checks[f"{modality}_prompt_clip_drop_at_most_0.01"] = (
            metrics["int8dq_quality"]["clip_prompt"]["mean"] >= metrics["int8wo_quality"]["clip_prompt"]["mean"] - 0.01
        )

    if "i2v" in modalities:
        metrics = aggregate["i2v"]
        checks["i2v_adjacent_lpips_within_10pct_of_int8wo"] = (
            metrics["int8dq_temporal"]["adjacent_lpips"]["mean"]
            <= 1.10 * metrics["int8wo_temporal"]["adjacent_lpips"]["mean"]
        )
        checks["i2v_condition_clip_drop_at_most_0.01"] = (
            metrics["int8dq_quality"]["clip_condition"]["mean"]
            >= metrics["int8wo_quality"]["clip_condition"]["mean"] - 0.01
        )
        for key in ("flow_magnitude", "flow_roughness"):
            dq_value = metrics["int8dq_temporal"][key]["mean"]
            wo_value = metrics["int8wo_temporal"][key]["mean"]
            checks[f"i2v_{key}_within_20pct_of_int8wo"] = 0.80 * wo_value <= dq_value <= 1.20 * wo_value

    return {
        "criteria": {
            "prompt_alignment": "DQ CLIP prompt score may trail WO by at most 0.01",
            "i2v_temporal": "DQ adjacent LPIPS <= 1.10 * WO; flow magnitude/roughness within 20% of WO",
            "i2v_conditioning": "DQ first-frame CLIP similarity to condition may trail WO by at most 0.01",
            "paired_reference_metrics": "LPIPS, SSIM, and PSNR versus BF16 are diagnostic, not pass/fail",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def evaluate(
    root: Path,
    seeds: Sequence[int],
    conditioning_image: Path | None,
    modalities: Sequence[str],
    t2i_prompt: str,
    i2v_prompt: str,
) -> dict[str, Any]:
    perceptual_model = lpips.LPIPS(net="alex", verbose=False).eval()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
    )
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    report: dict[str, Any] = {"seeds": list(seeds), "modalities": list(modalities)}

    if "t2i" in modalities:
        report["t2i"] = {}
        for seed in seeds:
            images = {mode: _read_image(root / f"t2i_{mode}" / "t2i" / str(seed) / "vision.jpg") for mode in MODES}
            seed_metrics: dict[str, Any] = {}
            for left, right in PAIR_NAMES:
                seed_metrics[f"{left}_vs_{right}"] = _pair_metrics(perceptual_model, images[left], images[right])
            for mode in MODES:
                seed_metrics[f"{mode}_quality"] = {
                    "clip_prompt": _clip_prompt_score(
                        clip_model, clip_preprocess, clip_tokenizer, [images[mode]], t2i_prompt
                    )
                }
            report["t2i"][str(seed)] = seed_metrics

    if "i2v" in modalities:
        report["i2v"] = {}
        condition = _read_image(conditioning_image) if conditioning_image else None
        for seed in seeds:
            videos = {mode: _read_video(root / f"i2v_{mode}" / "i2v" / str(seed) / "vision.mp4") for mode in MODES}
            frame_count = min(len(frames) for frames in videos.values())
            videos = {mode: frames[:frame_count] for mode, frames in videos.items()}
            seed_metrics = {}
            for left, right in PAIR_NAMES:
                rows = [_pair_metrics(perceptual_model, videos[left][i], videos[right][i]) for i in range(frame_count)]
                seed_metrics[f"{left}_vs_{right}"] = _mean_metrics(rows)
            for mode in MODES:
                seed_metrics[f"{mode}_temporal"] = _temporal_metrics(perceptual_model, videos[mode])
                quality = {
                    "clip_prompt": _clip_prompt_score(
                        clip_model, clip_preprocess, clip_tokenizer, videos[mode], i2v_prompt
                    )
                }
                if condition is not None:
                    resized_condition = cv2.resize(condition, (videos[mode][0].shape[1], videos[mode][0].shape[0]))
                    seed_metrics[f"{mode}_condition"] = _pair_metrics(
                        perceptual_model, videos[mode][0], resized_condition
                    )
                    quality["clip_condition"] = _clip_image_similarity(
                        clip_model, clip_preprocess, videos[mode][0], resized_condition
                    )
                seed_metrics[f"{mode}_quality"] = quality
            report["i2v"][str(seed)] = seed_metrics

    report["aggregate"] = {modality: _aggregate(report[modality]) for modality in modalities}
    report["promotion_gate"] = _promotion_gate(report["aggregate"], modalities)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--modalities", nargs="+", choices=("t2i", "i2v"), default=["t2i", "i2v"])
    parser.add_argument("--conditioning-image", type=Path)
    parser.add_argument("--t2i-input", type=Path, default=Path("inputs/omni/t2i.json"))
    parser.add_argument("--i2v-input", type=Path, default=Path("inputs/omni/i2v.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    t2i_prompt = json.loads(args.t2i_input.read_text())["extra"]["prompt"]
    i2v_prompt = json.loads(args.i2v_input.read_text())["extra"]["prompt"]
    report = evaluate(
        args.root,
        args.seeds,
        args.conditioning_image,
        args.modalities,
        t2i_prompt,
        i2v_prompt,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
