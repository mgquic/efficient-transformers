#!/usr/bin/env python3
"""Export and compile fixed-shape LLaDA-Image components for QAIC."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import torch


LOGGER = logging.getLogger("llada_image_components")
DEFAULT_MODEL = Path("/home/madhgupt/.cache/huggingface/models--inclusionAI--LLaDA-Image-Turbo/snapshots/f4afc52d925bbac4e22a1c947111fc1f127e37e5")
DEFAULT_QEFF_HOME = Path(".qeff-llada-image-components")


def _load_component(model_dir: Path, component: str):
    from QEfficient.diffusers.models.transformers.transformer_llada_image import (
        LLaDAImageQueryFormerModel,
        LLaDAImageSigVQModel,
        LLaDAImageTextProjectionModel,
    )

    if component == "queryformer":
        return LLaDAImageQueryFormerModel.from_pretrained(model_dir / "queryformer", torch_dtype=torch.float32)
    if component == "text_projector":
        return LLaDAImageTextProjectionModel.from_pretrained(model_dir / "text_projection", torch_dtype=torch.float32)
    if component == "sigvq_tokens":
        return LLaDAImageSigVQModel.from_pretrained(model_dir / "sigvq", torch_dtype=torch.float32)
    if component == "sigvq_image":
        return LLaDAImageSigVQModel.from_pretrained(model_dir / "sigvq", torch_dtype=torch.float32)
    raise ValueError(f"Unsupported component: {component}")


def _run_component(args: argparse.Namespace, component: str) -> None:
    from QEfficient.diffusers.pipelines.llada_image.component_module import QEffLLaDAImageComponent

    model = _load_component(args.model_path, component).eval()
    sequence_length = args.queryformer_sequence_length if component == "queryformer" else args.sequence_length
    adapter = QEffLLaDAImageComponent(
        model,
        component=component,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    export_dir = args.qeff_home / f"export-{component}-static-v2-s{sequence_length}-h{args.image_height}-w{args.image_width}"
    compile_dir = args.qeff_home / f"compile-{component}-static-v2-s{sequence_length}-h{args.image_height}-w{args.image_width}"
    LOGGER.info("%s export directory: %s", component, export_dir.resolve())
    LOGGER.info("%s compile directory: %s", component, compile_dir.resolve())
    start = time.monotonic()
    onnx_path = adapter.export(
        export_dir=export_dir,
        batch_size=1,
        sequence_length=sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
        offload_pt_weights=False,
        use_onnx_subfunctions=False,
    )
    LOGGER.info("%s ONNX: %s", component, Path(onnx_path).resolve())
    LOGGER.info("%s export duration: %.2f seconds", component, time.monotonic() - start)

    start = time.monotonic()
    qpc_path = adapter.compile(
        onnx_path=onnx_path,
        compile_dir=compile_dir,
        batch_size=1,
        sequence_length=sequence_length,
        image_height=args.image_height,
        image_width=args.image_width,
        aic_hw_version=args.aic_hw_version,
        aic_num_cores=args.aic_num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        use_onnx_subfunctions=False,
    )
    LOGGER.info("%s QPC: %s", component, Path(qpc_path).resolve())
    LOGGER.info("%s compile duration: %.2f seconds", component, time.monotonic() - start)
    del adapter, model
    gc.collect()


def _run_vae_decoder(args: argparse.Namespace) -> None:
    from diffusers.models import AutoencoderKLFlux2
    from QEfficient.diffusers.pipelines.llada_image import QEffLLaDAImageVAEDecoder

    model = AutoencoderKLFlux2.from_pretrained(args.model_path / "vae", torch_dtype=torch.float32)
    adapter = QEffLLaDAImageVAEDecoder(model)
    export_dir = args.qeff_home / f"export-vae-decoder-h{args.image_height}-w{args.image_width}"
    compile_dir = args.qeff_home / f"compile-vae-decoder-h{args.image_height}-w{args.image_width}"
    latent_height = args.image_height // 8
    latent_width = args.image_width // 8
    LOGGER.info("vae_decoder export directory: %s", export_dir.resolve())
    LOGGER.info("vae_decoder compile directory: %s", compile_dir.resolve())

    start = time.monotonic()
    inputs, dynamic_axes, output_names = adapter.get_onnx_params(
        latent_height=latent_height,
        latent_width=latent_width,
    )
    onnx_path = adapter.export(
        inputs=inputs,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_dir=str(export_dir),
        export_kwargs={"offload_pt_weights": False},
    )
    LOGGER.info("vae_decoder ONNX: %s", Path(onnx_path).resolve())
    LOGGER.info("vae_decoder export duration: %.2f seconds", time.monotonic() - start)

    start = time.monotonic()
    qpc_path = adapter.compile(
        onnx_path=onnx_path,
        compile_dir=str(compile_dir),
        specializations=[
            {
                "batch_size": 1,
                "channels": 32,
                "latent_height": latent_height,
                "latent_width": latent_width,
            }
        ],
        aic_hw_version=args.aic_hw_version,
        aic_num_cores=args.aic_num_cores,
        convert_to_fp16=True,
        aic_enable_depth_first=True,
    )
    LOGGER.info("vae_decoder QPC: %s", Path(qpc_path).resolve())
    LOGGER.info("vae_decoder compile duration: %.2f seconds", time.monotonic() - start)
    del adapter, model
    gc.collect()


def _run_llada2(args: argparse.Namespace, *, output_logits: bool) -> None:
    from QEfficient import QEFFAutoModelForCausalLM

    model_dir = args.model_path / "text_encoder"
    variant = "vq" if output_logits else "text"
    num_devices = args.llada2_num_devices if output_logits else args.llada2_text_num_devices
    device_suffix = f"-{num_devices}d" if num_devices > 1 else ""
    export_dir = args.qeff_home / f"export-llada2-{variant}-s{args.sequence_length}{device_suffix}"
    compile_dir = args.qeff_home / f"compile-llada2-{variant}-s{args.sequence_length}{device_suffix}"
    LOGGER.info("llada2 text model: %s", model_dir.resolve())
    LOGGER.info("llada2 export directory: %s", export_dir.resolve())
    LOGGER.info("llada2 compile directory: %s", compile_dir.resolve())
    start = time.monotonic()
    adapter = QEFFAutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        llada2_output_logits=output_logits,
    )
    LOGGER.info("LLaDA2 model load and wrapper initialization: %.2f seconds", time.monotonic() - start)
    start = time.monotonic()
    onnx_path = adapter.export(
        export_dir=export_dir,
        batch_size=1,
        sequence_length=args.sequence_length,
        offload_pt_weights=False,
        aic_num_cores=args.aic_num_cores,
        num_devices=num_devices,
        use_onnx_subfunctions=False,
    )
    LOGGER.info("llada2 ONNX: %s", Path(onnx_path).resolve())
    LOGGER.info("llada2 export duration: %.2f seconds", time.monotonic() - start)
    start = time.monotonic()
    qpc_path = adapter.compile(
        onnx_path=onnx_path,
        compile_dir=compile_dir,
        batch_size=1,
        sequence_length=args.sequence_length,
        num_devices=num_devices,
        aic_hw_version=args.aic_hw_version,
        aic_num_cores=args.aic_num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        use_onnx_subfunctions=False,
    )
    LOGGER.info("llada2_%s QPC: %s", variant, Path(qpc_path).resolve())
    LOGGER.info("llada2_%s compile duration: %.2f seconds", variant, time.monotonic() - start)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--qeff-home", type=Path, default=DEFAULT_QEFF_HOME)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=(
            "queryformer",
            "text_projector",
            "sigvq_tokens",
            "sigvq_image",
            "vae_decoder",
            "llada2",
            "llada2_vq",
        ),
        default=("queryformer", "text_projector", "sigvq_tokens", "sigvq_image"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument(
        "--queryformer-sequence-length",
        type=int,
        default=256,
        help="QueryFormer input length; text projector and LLaDA2 retain --sequence-length.",
    )
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--aic-hw-version", choices=("ai100", "ai200"), default="ai100")
    parser.add_argument("--aic-num-cores", type=int, default=16)
    parser.add_argument(
        "--mxfp6-matmul",
        action="store_true",
        help="Compress constant MatMul weights to MXFP6 while retaining FP16 activations.",
    )
    parser.add_argument(
        "--llada2-num-devices",
        type=int,
        default=2,
        help="Number of AI devices for the VQ LLaDA2 MoE graph.",
    )
    parser.add_argument(
        "--llada2-text-num-devices",
        type=int,
        default=1,
        help="Number of AI devices for the text-mode LLaDA2 MoE graph.",
    )
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    os.environ["HF_HUB_CACHE"] = str(Path(os.environ.get("HF_HUB_CACHE", "/home/madhgupt/.cache/huggingface")).resolve())
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("QEFF_HOME", str(args.qeff_home.resolve()))
    args.model_path = args.model_path.expanduser().resolve()
    args.qeff_home = args.qeff_home.expanduser().resolve()
    args.qeff_home.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)

    for component in args.components:
        if component in {"llada2", "llada2_vq"}:
            _run_llada2(args, output_logits=component == "llada2_vq")
        elif component == "vae_decoder":
            _run_vae_decoder(args)
        else:
            _run_component(args, component)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
