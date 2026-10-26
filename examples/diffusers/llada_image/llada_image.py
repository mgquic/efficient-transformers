#!/usr/bin/env python3
"""Flux-style LLaDA-Image export, compile, and inference example.

The QEfficient pipeline owns export/compile configuration and runtime QPC
selection. This example only supplies the model, generation arguments, and
deployment configuration.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = Path("/home/madhgupt/.cache/huggingface")
DEFAULT_CONFIG = REPO_ROOT / "examples/diffusers/llada_image/llada_image_config.json"
COMPILE_CONFIG = Path(os.environ.get("LLADA_CONFIG", str(DEFAULT_CONFIG))).expanduser().resolve()
DEFAULT_WORK_ROOT = REPO_ROOT / "e2e"
MODEL_REPO = "inclusionAI/LLaDA-Image-Turbo"
MODEL_REVISION = "f4afc52d925bbac4e22a1c947111fc1f127e37e5"

HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", str(DEFAULT_CACHE))).expanduser().resolve()


def _find_local_model_snapshot() -> Path | None:
    for cache_root in dict.fromkeys((HF_HUB_CACHE, DEFAULT_CACHE)):
        snapshot = (
            cache_root
            / "models--inclusionAI--LLaDA-Image-Turbo"
            / "snapshots"
            / MODEL_REVISION
        )
        if (snapshot / "model_index.json").is_file():
            return snapshot
    return None


MODEL_PATH = os.environ.get("LLADA_MODEL_ID")
if MODEL_PATH is None:
    local_model = _find_local_model_snapshot()
    if local_model is not None:
        MODEL_PATH = str(local_model)
        HF_HUB_CACHE = local_model.parents[2]
    else:
        MODEL_PATH = MODEL_REPO
sys.path.insert(0, str(REPO_ROOT))
os.environ["PATH"] = os.pathsep.join(filter(None, ["/opt/qti-aic/exec", os.environ.get("PATH", "")]))
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(
        None,
        [
            str(REPO_ROOT),
            "/opt/qti-aic/dev/lib/x86_64",
            "/opt/qti-aic/dev/python",
            os.environ.get("PYTHONPATH", ""),
        ],
    )
)
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
    filter(None, ["/opt/qti-aic/dev/lib/x86_64", os.environ.get("LD_LIBRARY_PATH", "")])
)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partial-layers",
        action="store_true",
        help=(
            "Keep only first/last repeated blocks in the fused encoder, DiT, "
            "and VAE decoder for export/compile latency tests."
        ),
    )
    return parser.parse_args()


def _prepare_work_root(partial_layers: bool) -> tuple[Path, Path]:
    default_root = DEFAULT_WORK_ROOT / "partial_2layer" if partial_layers else DEFAULT_WORK_ROOT
    work_root = Path(os.environ.get("LLADA_WORK_ROOT", str(default_root))).expanduser().resolve()
    output = Path(
        os.environ.get(
            "LLADA_OUTPUT",
            str(work_root / "llada_image_1024.png"),
        )
    ).expanduser().resolve()

    os.environ["QEFF_HOME"] = str(work_root)
    if work_root.exists() and any(work_root.iterdir()) and not (work_root / "manifest.json").is_file():
        raise RuntimeError(
            f"Refusing to reuse existing e2e artifacts: {work_root}. "
            "Use a new empty directory or remove the incomplete directory explicitly."
        )
    work_root.mkdir(parents=True, exist_ok=True)
    return work_root, output


def main() -> None:
    args = _parse_args()
    work_root, output_path = _prepare_work_root(args.partial_layers)
    import torch

    from QEfficient import QEffLLaDAImagePipeline

    pipeline = QEffLLaDAImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        composite_text_encoder_implementation="default",
    )

    if args.partial_layers:
        from QEfficient.diffusers.pipelines.llada_image.partial_layers import (
            truncate_llada_image_pipeline,
        )

        partial_report = truncate_llada_image_pipeline(pipeline)
        print(f"Using partial first/last-layer export mode; report: {partial_report}")

    result = pipeline(
        prompt="A man drawing potrait of Mahatma Gandhi",
        height=int(os.environ.get("LLADA_HEIGHT", "1024")),
        width=int(os.environ.get("LLADA_WIDTH", "1024")),
        num_inference_steps=int(os.environ.get("LLADA_STEPS", "4")),
        guidance_scale=float(os.environ.get("LLADA_GUIDANCE_SCALE", "1.0")),
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(42),
        custom_config_path=str(COMPILE_CONFIG),
        work_root=str(work_root),
        parallel_compile=os.environ.get("LLADA_PARALLEL_COMPILE", "0") == "1",
        use_onnx_subfunctions=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved image to {output_path}")
    print(getattr(pipeline, "last_timings", {}))


if __name__ == "__main__":
    main()
