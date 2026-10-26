#!/usr/bin/env python3
"""Run LLaDA-Image with DiT/VAE graphs on AI100 and the remaining stages on CPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = Path("/home/madhgupt/.cache/huggingface")
DEFAULT_MODEL = (
    DEFAULT_CACHE
    / "models--inclusionAI--LLaDA-Image-Turbo"
    / "snapshots"
    / "f4afc52d925bbac4e22a1c947111fc1f127e37e5"
)
DEFAULT_QEFF_HOME = REPO_ROOT / ".qeff-llada-image"
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

logger = logging.getLogger("llada_image_aic_pipeline")


@dataclass
class LLaDAImageAICResult:
    """Pipeline-like result containing generated images and stage timings."""

    images: list[Image.Image]
    timings: dict[str, float]


def _configure_logging(log_path: Path | None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


@contextmanager
def _timed(timings: dict[str, float], name: str):
    start = time.perf_counter()
    logger.info("START %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        logger.info("END %s | %.2f seconds", name, elapsed)


def _register_llada_rope_compatibility() -> None:
    """Register the legacy RoPE name expected by the remote LLaDA text encoder."""
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def compute_default_rope_parameters(config, device=None, seq_len=None, **_kwargs):
        del seq_len
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


def _resolve_qpc(explicit_path: Path | None, compile_dir: Path, operation: str) -> Path:
    if explicit_path is not None:
        qpc_path = explicit_path.expanduser().resolve()
        if not qpc_path.is_dir():
            raise FileNotFoundError(f"{operation} QPC does not exist: {qpc_path}")
        return qpc_path

    qpcs = sorted(path for path in compile_dir.glob("qpc-*/qpc") if path.is_dir())
    if not qpcs:
        raise FileNotFoundError(
            f"No {operation} QPC found under {compile_dir}. "
            f"Pass --{operation}-qpc-path or compile the {operation} graph first."
        )
    if len(qpcs) > 1:
        choices = ", ".join(str(path) for path in qpcs)
        raise RuntimeError(f"Multiple {operation} QPCs found; pass an explicit path: {choices}")
    return qpcs[0].resolve()


def _resolve_model_dir(model_path: str | Path, cache_dir: Path) -> Path:
    model_dir = Path(model_path).expanduser()
    if model_dir.is_dir():
        return model_dir.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model_path), cache_dir=str(cache_dir))).resolve()


def _load_vae_adapter(model_dir: Path, operation: str):
    from diffusers.models import AutoencoderKLFlux2
    from QEfficient.diffusers.pipelines.llada_image import (
        QEffLLaDAImageVAEDecoder,
        QEffLLaDAImageVAEEncoder,
    )

    vae = AutoencoderKLFlux2.from_pretrained(str(model_dir / "vae"), torch_dtype=torch.float32)
    adapter_class = QEffLLaDAImageVAEEncoder if operation == "encoder" else QEffLLaDAImageVAEDecoder
    return vae, adapter_class(vae)


def _export_and_compile_vae(
    model_dir: Path,
    operation: str,
    height: int,
    width: int,
    export_dir: Path,
    compile_dir: Path,
    aic_hw_version: str | None,
    aic_num_cores: int | None,
    timings: dict[str, float],
) -> Path:
    import QEfficient.utils.constants as constants

    vae, adapter = _load_vae_adapter(model_dir, operation)
    if operation == "encoder":
        with _timed(timings, "export VAE encoder to ONNX"):
            example_inputs, dynamic_axes, output_names = adapter.get_onnx_params(height, width)
            onnx_path = adapter.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=str(export_dir),
                export_kwargs={"offload_pt_weights": False},
            )
        specializations = [{"batch_size": 1, "height": height, "width": width}]
    else:
        with _timed(timings, "export VAE decoder to ONNX"):
            example_inputs, dynamic_axes, output_names = adapter.get_onnx_params(
                latent_height=height // 8,
                latent_width=width // 8,
            )
            onnx_path = adapter.export(
                inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=str(export_dir),
                export_kwargs={"offload_pt_weights": False},
            )
        specializations = [
            {
                "batch_size": 1,
                "channels": 32,
                "latent_height": height // 8,
                "latent_width": width // 8,
            }
        ]

    hardware = aic_hw_version or constants.DEFAULT_AIC_HW_VERSION
    cores = aic_num_cores or constants.DEFAULT_AIC_NUM_CORES
    with _timed(timings, f"compile VAE {operation} for {hardware}"):
        adapter.compile(
            specializations=specializations,
            onnx_path=str(onnx_path),
            compile_dir=str(compile_dir),
            aic_hw_version=hardware,
            aic_num_cores=cores,
            convert_to_fp16=True,
            aic_enable_depth_first=True,
        )
    qpc_path = Path(adapter.qpc_path or compile_dir / "qpc")
    logger.info("Compiled %s QPC: %s", operation, qpc_path.resolve())
    del vae
    return qpc_path.resolve()


def _export_and_compile_dit(
    transformer,
    *,
    mode: str,
    height: int,
    width: int,
    caption_length: int,
    semantic_length: int,
    export_dir: Path,
    compile_dir: Path,
    aic_hw_version: str | None,
    aic_num_cores: int | None,
    use_onnx_subfunctions: bool,
    timings: dict[str, float],
) -> Path:
    """Export and compile one fixed-shape LLaDA-Image DiT graph."""
    from QEfficient.diffusers.pipelines.llada_image import QEffLLaDAImageTransformer
    import QEfficient.utils.constants as constants

    encoder = QEffLLaDAImageTransformer(
        transformer,
        mode=mode,
        latent_height=height // 16,
        latent_width=width // 16,
        caption_length=caption_length,
        semantic_length=semantic_length,
    )
    with _timed(timings, f"export LLaDA-Image {mode} DiT graph to ONNX"):
        onnx_path = encoder.export(
            export_dir=export_dir,
            batch_size=1,
            offload_pt_weights=False,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
    hardware = aic_hw_version or constants.DEFAULT_AIC_HW_VERSION
    cores = aic_num_cores or constants.DEFAULT_AIC_NUM_CORES
    with _timed(timings, f"compile LLaDA-Image {mode} DiT graph for {hardware}"):
        qpc_path = encoder.compile(
            onnx_path=onnx_path,
            compile_dir=str(compile_dir),
            batch_size=1,
            aic_hw_version=hardware,
            aic_num_cores=cores,
            convert_to_fp16=True,
        )
    if qpc_path is None:
        raise RuntimeError(f"LLaDA-Image {mode} DiT compilation did not return a QPC path")
    logger.info("Compiled LLaDA-Image %s DiT QPC: %s", mode, Path(qpc_path).resolve())
    return Path(qpc_path).resolve()


def _semantic_length(mode: str, height: int, width: int) -> int:
    if mode == "vq":
        frontend_scale = max(max(height, width) / 512, 1.0)
        return int(height / frontend_scale) // 16 * (int(width / frontend_scale) // 16)
    if mode == "editing":
        return (height // 32) * (width // 32)
    return 0


def _reduce_dit_layers(transformer, layer_count: int) -> None:
    """Keep the first ``layer_count`` blocks for a Flux-style export smoke test."""
    if layer_count < 1:
        raise ValueError("dit_layer_count must be at least 1")

    block_groups = (
        "noise_refiner",
        "context_refiner",
        "sigvq_refiner",
        "layers",
    )
    available = {name: len(getattr(transformer, name)) for name in block_groups}
    if any(layer_count > count for count in available.values()):
        raise ValueError(f"dit_layer_count={layer_count} exceeds available DiT blocks: {available}")
    for name in block_groups:
        setattr(transformer, name, torch.nn.ModuleList(list(getattr(transformer, name)[:layer_count])))
    transformer.config.n_layers = layer_count
    transformer.config.n_refiner_layers = layer_count


def _device_ids(value: str | Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    return [int(item) for item in value]


def _component_device_ids(value: str | Sequence[int] | None, component: str) -> list[int] | None:
    """Select the device group for each concurrently resident QPC.

    A single device ID preserves the existing behavior. With multiple IDs, the
    graphs are spread across cards because the LLaDA2 graphs can consume most of
    one card's DRAM. The VQ LLaDA2 QPC is a two-device MDP graph.
    """
    device_list = _device_ids(value)
    if device_list is None or len(device_list) == 1:
        return device_list
    allocations = {
        "composite_text_encoder": (0, 1),
        "queryformer": (0, 1),
        "llada2": (1, 1),
        "text_projector": (2, 1),
        "llada2_vq": (3, 2),
        "sigvq_tokens": (5, 1),
        "dit": (6, 1),
        "decoder": (7, 1),
        "encoder": (8, 1),
    }
    if component not in allocations:
        raise ValueError(f"Unknown QPC component: {component}")
    offset, count = allocations[component]
    if offset + count > len(device_list):
        raise ValueError(
            f"--device-ids supplied {len(device_list)} IDs, but component {component!r} needs "
            f"positions {offset} through {offset + count - 1}"
        )
    return device_list[offset : offset + count]


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, channels * 4, height // 2, width // 2)


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = latents.shape
    latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, channels // 4, height * 2, width * 2)


def _normalize_source_latents(raw_latents: torch.Tensor, vae) -> torch.Tensor:
    patchified = _patchify_latents(raw_latents.float())
    latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(patchified)
    latent_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(patchified)
    return (patchified - latent_mean) / latent_std


def _denormalize_decoder_latents(patchified_latents: torch.Tensor, vae) -> torch.Tensor:
    latent_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(patchified_latents)
    latent_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(patchified_latents)
    return _unpatchify_latents(patchified_latents.float() * latent_std + latent_mean)


def _run_aic_encoder(
    qpc_path: Path,
    pixels: torch.Tensor,
    height: int,
    width: int,
    device_ids: list[int] | None,
    timings: dict[str, float],
) -> torch.Tensor:
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    output_shape = (pixels.shape[0], 32, height // 8, width // 8)
    session = None
    try:
        with _timed(timings, "AIC VAE encoder session initialization"):
            session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
            session.set_buffers({"latents": np.zeros(output_shape, dtype=np.float32)})
        with _timed(timings, "AIC VAE encode"):
            output = session.run({"image": pixels.numpy()})["latents"]
    finally:
        if session is not None:
            session.deactivate()

    latents = torch.from_numpy(np.array(output, copy=True)).float()
    if tuple(latents.shape) != output_shape:
        raise RuntimeError(f"AIC encoder returned {tuple(latents.shape)}, expected {output_shape}")
    if not torch.isfinite(latents).all():
        raise RuntimeError("AIC VAE encoder returned non-finite latents")
    return latents


def _run_aic_decoder(
    qpc_path: Path,
    decoder_latents: torch.Tensor,
    pipeline,
    height: int,
    width: int,
    device_ids: list[int] | None,
    timings: dict[str, float],
) -> Image.Image:
    from diffusers.image_processor import VaeImageProcessor
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    output_shape = (decoder_latents.shape[0], 3, height, width)
    session = None
    try:
        with _timed(timings, "AIC VAE decoder session initialization"):
            session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
            session.set_buffers({"sample": np.zeros(output_shape, dtype=np.float32)})
        with _timed(timings, "AIC VAE decode"):
            output = session.run({"latent_sample": decoder_latents.numpy()})["sample"]
    finally:
        if session is not None:
            session.deactivate()

    output = np.array(output, copy=True)
    if not np.isfinite(output).all():
        raise RuntimeError("AIC VAE decoder returned non-finite image output")
    with _timed(timings, "AIC image post-processing"):
        return VaeImageProcessor(vae_scale_factor=pipeline.vae_scale_factor).postprocess(
            torch.from_numpy(output), output_type="pil"
        )[0]


def pipe(
    prompt: str,
    *,
    generation_mode: str = "text",
    image_path: str | Path | None = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    negative_prompt: str | None = None,
    seed: int = 42,
    cpu_dtype: torch.dtype = torch.float32,
    model_path: str | Path = DEFAULT_MODEL,
    cache_dir: str | Path = DEFAULT_CACHE,
    encoder_qpc_path: str | Path | None = None,
    decoder_qpc_path: str | Path | None = None,
    encoder_compile_dir: str | Path | None = None,
    decoder_compile_dir: str | Path | None = None,
    encoder_export_dir: str | Path | None = None,
    decoder_export_dir: str | Path | None = None,
    dit_qpc_path: str | Path | None = None,
    composite_text_encoder_qpc_path: str | Path | None = None,
    queryformer_qpc_path: str | Path | None = None,
    text_projector_qpc_path: str | Path | None = None,
    llada2_text_qpc_path: str | Path | None = None,
    llada2_vq_qpc_path: str | Path | None = None,
    sigvq_tokens_qpc_path: str | Path | None = None,
    dit_compile_dir: str | Path | None = None,
    dit_export_dir: str | Path | None = None,
    dit_layer_count: int | None = None,
    dit_use_onnx_subfunctions: bool = True,
    dit_device_ids: str | Sequence[int] | None = None,
    composite_text_encoder_device_ids: str | Sequence[int] | None = None,
    stage_device_sessions: bool = False,
    aic_hw_version: str | None = None,
    aic_num_cores: int | None = None,
    device_ids: str | Sequence[int] | None = None,
    llada2_device_ids: str | Sequence[int] | None = None,
    output_path: str | Path | None = None,
    log_path: str | Path | None = None,
    max_sequence_length: int = 256,
) -> LLaDAImageAICResult:
    """Run LLaDA-Image with selected QPC graphs on AI100 and the remaining stages on CPU.

    ``generation_mode`` can be ``text``, ``vq``, or ``editing``. Editing requires
    ``image_path`` and uses the AI100 encoder; all modes use the AI100 decoder.
    The QPCs must be compiled for the requested resolution and batch size.
    """
    if generation_mode == "editing" and image_path is None:
        raise ValueError("image_path is required when generation_mode='editing'")
    if generation_mode in {"text", "vq"} and image_path is not None:
        raise ValueError(f"image_path must be omitted when generation_mode='{generation_mode}'")
    if cpu_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("cpu_dtype must be torch.float32 or torch.bfloat16")

    cache_dir = Path(cache_dir).expanduser().resolve()
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ.setdefault("QEFF_HOME", str(DEFAULT_QEFF_HOME))
    qeff_home = Path(os.environ["QEFF_HOME"]).expanduser().resolve()
    encoder_compile_dir = Path(encoder_compile_dir or qeff_home / f"clean-{height}-compile-encoder").resolve()
    decoder_compile_dir = Path(decoder_compile_dir or qeff_home / f"clean-{height}-compile-decoder").resolve()
    encoder_export_dir = Path(encoder_export_dir or qeff_home / f"clean-{height}-export-encoder").resolve()
    decoder_export_dir = Path(decoder_export_dir or qeff_home / f"clean-{height}-export-decoder").resolve()
    dit_compile_dir = Path(dit_compile_dir or qeff_home / f"dit-{height}-compile-{generation_mode}").resolve()
    dit_export_dir = Path(dit_export_dir or qeff_home / f"dit-{height}-export-{generation_mode}").resolve()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _configure_logging(Path(log_path).expanduser() if log_path is not None else None)

    model_dir = _resolve_model_dir(model_path, cache_dir)
    from QEfficient.diffusers.pipelines.llada_image import QEffLLaDAImagePipeline

    timings: dict[str, float] = {}
    resolved_decoder_qpc = None
    try:
        resolved_decoder_qpc = _resolve_qpc(
            Path(decoder_qpc_path) if decoder_qpc_path is not None else None,
            decoder_compile_dir,
            "decoder",
        )
    except FileNotFoundError:
        logger.info("Decoder QPC not found; exporting and compiling it")
        resolved_decoder_qpc = _export_and_compile_vae(
            model_dir,
            "decoder",
            height,
            width,
            decoder_export_dir,
            decoder_compile_dir,
            aic_hw_version,
            aic_num_cores,
            timings,
        )
    resolved_encoder_qpc = None
    if generation_mode == "editing":
        try:
            resolved_encoder_qpc = _resolve_qpc(
                Path(encoder_qpc_path) if encoder_qpc_path is not None else None,
                encoder_compile_dir,
                "encoder",
            )
        except FileNotFoundError:
            logger.info("Encoder QPC not found; exporting and compiling it")
            resolved_encoder_qpc = _export_and_compile_vae(
                model_dir,
                "encoder",
                height,
                width,
                encoder_export_dir,
                encoder_compile_dir,
                aic_hw_version,
                aic_num_cores,
                timings,
            )

    logger.info("generation_mode=%s resolution=%dx%d cpu_dtype=%s", generation_mode, height, width, cpu_dtype)
    logger.info("model_path=%s", model_path)
    logger.info("decoder_qpc=%s", resolved_decoder_qpc)
    if resolved_encoder_qpc is not None:
        logger.info("encoder_qpc=%s", resolved_encoder_qpc)

    _register_llada_rope_compatibility()
    with _timed(timings, "CPU pipeline initialization"):
        qeff_pipeline = QEffLLaDAImagePipeline.from_pretrained(model_dir, torch_dtype=cpu_dtype)
        pipeline = qeff_pipeline.model

    if dit_layer_count is not None:
        _reduce_dit_layers(pipeline.transformer, dit_layer_count)
        logger.warning(
            "Using a reduced LLaDA-Image DiT with %d block(s) in each refiner/main stack; "
            "this is an export/compile smoke configuration, not the full-quality model",
            dit_layer_count,
        )

    caption_length = max_sequence_length + int(getattr(pipeline.queryformer.config, "num_queries", 256))
    dit_semantic_length = _semantic_length(generation_mode, height, width)
    try:
        resolved_dit_qpc = _resolve_qpc(
            Path(dit_qpc_path) if dit_qpc_path is not None else None,
            dit_compile_dir,
            f"dit-{generation_mode}",
        )
    except FileNotFoundError:
        logger.info("LLaDA-Image %s DiT QPC not found; exporting and compiling it", generation_mode)
        resolved_dit_qpc = _export_and_compile_dit(
            pipeline.transformer,
            mode=generation_mode,
            height=height,
            width=width,
            caption_length=caption_length,
            semantic_length=dit_semantic_length,
            export_dir=dit_export_dir,
            compile_dir=dit_compile_dir,
            aic_hw_version=aic_hw_version,
            aic_num_cores=aic_num_cores,
            use_onnx_subfunctions=dit_use_onnx_subfunctions,
            timings=timings,
        )

    from QEfficient.diffusers.pipelines.llada_image.pipeline_module import LLaDAImageDenoisingAICSession

    if stage_device_sessions and generation_mode != "text":
        raise ValueError("stage_device_sessions currently supports generation_mode='text' only")
    if stage_device_sessions and guidance_scale > 1.0:
        raise ValueError("stage_device_sessions currently supports guidance_scale <= 1.0 only")
    resolved_dit_device_ids = (
        _device_ids(dit_device_ids)
        if dit_device_ids is not None
        else _component_device_ids(device_ids, "dit")
    )
    resolved_composite_text_encoder_device_ids = (
        _device_ids(composite_text_encoder_device_ids)
        if composite_text_encoder_device_ids is not None
        else _component_device_ids(device_ids, "composite_text_encoder")
    )
    dit_session = None
    if not stage_device_sessions:
        dit_session = LLaDAImageDenoisingAICSession(
            str(resolved_dit_qpc),
            mode=generation_mode,
            latent_shape=(1, pipeline.transformer.config.in_channels, height // 16, width // 16),
            caption_length=caption_length,
            semantic_length=dit_semantic_length,
            rope_config=pipeline.transformer.config,
            device_ids=resolved_dit_device_ids,
            timings=timings,
        )
    pipeline.dit_session = dit_session
    pipeline.component_timings = timings
    pipeline.stage_device_sessions = stage_device_sessions

    aic_sessions = [dit_session] if dit_session is not None else []
    device_id_list = _device_ids(device_ids)
    if device_id_list is not None:
        allocation = []
        if composite_text_encoder_qpc_path is not None:
            allocation.append(f"FusedTextEncoder={resolved_composite_text_encoder_device_ids}")
        else:
            if queryformer_qpc_path is not None:
                allocation.append(f"QueryFormer={_component_device_ids(device_ids, 'queryformer')}")
            if llada2_text_qpc_path is not None:
                llada2_devices = (
                    _device_ids(llada2_device_ids)
                    if llada2_device_ids is not None
                    else _component_device_ids(device_ids, "llada2")
                )
                allocation.append(f"LLaDA2Text={llada2_devices}")
            if text_projector_qpc_path is not None:
                allocation.append(f"TextProjector={_component_device_ids(device_ids, 'text_projector')}")
        if generation_mode == "vq" and llada2_vq_qpc_path is not None:
            allocation.append(f"LLaDA2VQ={_component_device_ids(device_ids, 'llada2_vq')}")
        if generation_mode == "vq" and sigvq_tokens_qpc_path is not None:
            allocation.append(f"SigVQTokens={_component_device_ids(device_ids, 'sigvq_tokens')}")
        if generation_mode == "editing" and resolved_encoder_qpc is not None:
            allocation.append(f"VAEEncoder={_component_device_ids(device_ids, 'encoder')}")
        allocation.extend(
            [
                f"DiT={resolved_dit_device_ids}",
                f"VAEDecoder={_component_device_ids(device_ids, 'decoder')}",
            ]
        )
        logger.info("AIC device allocation: %s", " ".join(allocation))
    if composite_text_encoder_qpc_path is not None:
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import (
            LLaDAImageCompositeTextEncoderAICSession,
        )

        composite_text_encoder_session = LLaDAImageCompositeTextEncoderAICSession(
            str(Path(composite_text_encoder_qpc_path).expanduser().resolve()),
            input_length=max_sequence_length,
            hidden_size=int(pipeline.text_encoder.config.hidden_size),
            output_length=max_sequence_length + int(getattr(pipeline.queryformer.config, "num_queries", 256)),
            projection_dim=int(pipeline.text_projection.config.projection_dim),
            num_queries=int(getattr(pipeline.queryformer.config, "num_queries", 256)),
            device_ids=resolved_composite_text_encoder_device_ids,
        )
        pipeline.composite_text_encoder_session = composite_text_encoder_session
        aic_sessions.append(composite_text_encoder_session)
        logger.info(
            "Fused text encoder (QueryFormer + LLaDA2 + text projector): AIC QPC=%s devices=%s",
            Path(composite_text_encoder_qpc_path).expanduser().resolve(),
            resolved_composite_text_encoder_device_ids,
        )
    elif queryformer_qpc_path is not None:
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import LLaDAImageQueryFormerAICSession

        queryformer_session = LLaDAImageQueryFormerAICSession(
            str(Path(queryformer_qpc_path).expanduser().resolve()),
            input_length=max_sequence_length,
            hidden_size=int(pipeline.queryformer.config.hidden_size),
            num_queries=int(pipeline.queryformer.config.num_queries),
            device_ids=_component_device_ids(device_ids, "queryformer"),
        )
        pipeline.queryformer_session = queryformer_session
        aic_sessions.append(queryformer_session)
        logger.info("QueryFormer: AIC QPC=%s", Path(queryformer_qpc_path).expanduser().resolve())
    elif composite_text_encoder_qpc_path is None:
        logger.info("QueryFormer: CPU PyTorch")

    if composite_text_encoder_qpc_path is None and llada2_text_qpc_path is not None:
        from QEfficient.transformers.llada2_utils import LLaDA2AICSession

        llada2_text_session = LLaDA2AICSession(
            str(Path(llada2_text_qpc_path).expanduser().resolve()),
            sequence_length=512,
            hidden_size=int(pipeline.text_encoder.config.hidden_size),
            device_ids=(
                _device_ids(llada2_device_ids)
                if llada2_device_ids is not None
                else _component_device_ids(device_ids, "llada2")
            ),
        )
        pipeline.llada2_text_session = llada2_text_session
        aic_sessions.append(llada2_text_session)
        logger.info("LLaDA2-MoE text encoder: AIC QPC=%s", Path(llada2_text_qpc_path).expanduser().resolve())
    elif composite_text_encoder_qpc_path is None:
        logger.info("LLaDA2-MoE text encoder: CPU PyTorch")

    if generation_mode == "vq" and llada2_vq_qpc_path is not None:
        from QEfficient.transformers.llada2_utils import LLaDA2AICSession

        llada2_vq_session = LLaDA2AICSession(
            str(Path(llada2_vq_qpc_path).expanduser().resolve()),
            sequence_length=512,
            hidden_size=int(pipeline.text_encoder.config.hidden_size),
            vocab_size=int(getattr(pipeline.text_encoder.config, "vocab_size", 173568)),
            vq=True,
            device_ids=_component_device_ids(device_ids, "llada2_vq"),
        )
        pipeline.llada2_vq_session = llada2_vq_session
        aic_sessions.append(llada2_vq_session)
        logger.info("LLaDA2-MoE VQ token generator: AIC QPC=%s", Path(llada2_vq_qpc_path).expanduser().resolve())
    elif generation_mode == "vq":
        logger.info("LLaDA2-MoE VQ token generator: CPU PyTorch")

    if composite_text_encoder_qpc_path is None and text_projector_qpc_path is not None:
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import LLaDAImageTextProjectorAICSession

        text_projection_session = LLaDAImageTextProjectorAICSession(
            str(Path(text_projector_qpc_path).expanduser().resolve()),
            sequence_length=512,
            hidden_size=int(pipeline.text_projection.config.hidden_size),
            projection_dim=int(pipeline.text_projection.config.projection_dim),
            device_ids=_component_device_ids(device_ids, "text_projector"),
            defer_activation=stage_device_sessions,
        )
        pipeline.text_projection_session = text_projection_session
        aic_sessions.append(text_projection_session)
        logger.info("Text projector: AIC QPC=%s", Path(text_projector_qpc_path).expanduser().resolve())
    elif composite_text_encoder_qpc_path is None:
        logger.info("Text projector: CPU PyTorch")

    if generation_mode == "vq" and sigvq_tokens_qpc_path is not None:
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import LLaDAImageSigVQTokensAICSession

        sigvq_tokens_session = LLaDAImageSigVQTokensAICSession(
            str(Path(sigvq_tokens_qpc_path).expanduser().resolve()),
            sequence_length=height // 16 * (width // 16),
            feature_dim=int(
                getattr(
                    pipeline.sigvq.config,
                    "semantic_embed_dim",
                    getattr(pipeline.sigvq.config, "hidden_size", 4096),
                )
            ),
            device_ids=_component_device_ids(device_ids, "sigvq_tokens"),
        )
        pipeline.sigvq_tokens_session = sigvq_tokens_session
        aic_sessions.append(sigvq_tokens_session)
        logger.info("SigVQ token encoder: AIC QPC=%s", Path(sigvq_tokens_qpc_path).expanduser().resolve())
    elif generation_mode == "vq":
        logger.info("SigVQ token encoder: CPU PyTorch")

    logger.info("Host placement: tokenizer, embedding lookup, masks, scheduler, latent transforms, image post-processing")
    logger.info("Text-mode SigVQ and VAE encoder: not used")
    logger.info("DiT QPC=%s", resolved_dit_qpc)

    image = None
    source_latents = None
    if image_path is not None:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB").copy()

    if generation_mode == "editing":
        with _timed(timings, "CPU image preprocessing"):
            pixels = pipeline.image_processor.preprocess(image, height=height, width=width).float().contiguous()
        raw_latents = _run_aic_encoder(
            resolved_encoder_qpc,
            pixels,
            height,
            width,
            _component_device_ids(device_ids, "encoder"),
            timings,
        )
        with _timed(timings, "CPU source latent normalization"):
            source_latents = _normalize_source_latents(raw_latents, pipeline.vae)

    precomputed_prompt = None
    if stage_device_sessions:
        do_classifier_free_guidance = guidance_scale > 1.0
        with _timed(timings, "AIC fused text conditioning + encoder teardown"):
            precomputed_prompt = pipeline.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=pipeline._execution_device,
            )
        for session_name in (
            "composite_text_encoder_session",
            "queryformer_session",
            "llada2_text_session",
            "text_projection_session",
        ):
            session = getattr(pipeline, session_name, None)
            if session is not None:
                session.close()
                setattr(pipeline, session_name, None)
        with _timed(timings, "AIC DiT QPC session initialization"):
            dit_session = LLaDAImageDenoisingAICSession(
                str(resolved_dit_qpc),
                mode=generation_mode,
                latent_shape=(1, pipeline.transformer.config.in_channels, height // 16, width // 16),
                caption_length=caption_length,
                semantic_length=dit_semantic_length,
                rope_config=pipeline.transformer.config,
                device_ids=resolved_dit_device_ids,
                timings=timings,
            )
        pipeline.dit_session = dit_session
        aic_sessions.append(dit_session)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with _timed(timings, "Denoising loop: host orchestration + AIC DiT"):
        pipeline_kwargs = dict(
            prompt=None if precomputed_prompt is not None else prompt,
            image=image,
            generation_mode=generation_mode,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            precomputed_source_latents=source_latents,
            max_sequence_length=max_sequence_length,
            output_type="latent",
        )
        if precomputed_prompt is not None:
            (
                pipeline_kwargs["prompt_embeds"],
                pipeline_kwargs["prompt_attention_mask"],
                pipeline_kwargs["negative_prompt_embeds"],
                pipeline_kwargs["negative_prompt_attention_mask"],
            ) = precomputed_prompt
            pipeline_kwargs["negative_prompt"] = None
        result = pipeline(**pipeline_kwargs)
    if stage_device_sessions and dit_session is not None:
        dit_session.close()
        pipeline.dit_session = None
    for name, elapsed in sorted(timings.items()):
        if name.startswith("component."):
            logger.info("COMPONENT_TIMING %s | %.2f seconds", name.removeprefix("component."), elapsed)

    with _timed(timings, "CPU decoder latent denormalization"):
        decoder_latents = _denormalize_decoder_latents(result.images.float(), pipeline.vae).contiguous()
    if not torch.isfinite(decoder_latents).all():
        raise RuntimeError("CPU denoising returned non-finite decoder latents")

    image = _run_aic_decoder(
        resolved_decoder_qpc,
        decoder_latents,
        pipeline,
        height,
        width,
        _component_device_ids(device_ids, "decoder"),
        timings,
    )
    if output_path is not None:
        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        logger.info("Saved image: %s", output_path.resolve())
    for session in reversed(aic_sessions):
        session.close()
    return LLaDAImageAICResult(images=[image], timings=timings)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--generation-mode", choices=("text", "vq", "editing"), default="text")
    parser.add_argument("--input-image", type=Path)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--encoder-qpc-path", type=Path)
    parser.add_argument("--decoder-qpc-path", type=Path)
    parser.add_argument("--encoder-compile-dir", type=Path)
    parser.add_argument("--decoder-compile-dir", type=Path)
    parser.add_argument("--encoder-export-dir", type=Path)
    parser.add_argument("--decoder-export-dir", type=Path)
    parser.add_argument("--dit-qpc-path", type=Path)
    parser.add_argument("--composite-text-encoder-qpc-path", type=Path)
    parser.add_argument("--queryformer-qpc-path", type=Path)
    parser.add_argument("--text-projector-qpc-path", type=Path)
    parser.add_argument("--llada2-text-qpc-path", type=Path)
    parser.add_argument("--llada2-vq-qpc-path", type=Path)
    parser.add_argument("--sigvq-tokens-qpc-path", type=Path)
    parser.add_argument("--dit-compile-dir", type=Path)
    parser.add_argument("--dit-export-dir", type=Path)
    parser.add_argument(
        "--dit-layer-count",
        type=int,
        help="Keep the first N blocks in every DiT stack for a Flux-style export smoke test.",
    )
    parser.add_argument(
        "--dit-use-onnx-subfunctions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export repeated DiT blocks as ONNX subfunctions (default: enabled).",
    )
    parser.add_argument("--dit-device-ids", help="Comma-separated device IDs for the DiT QPC.")
    parser.add_argument(
        "--composite-text-encoder-device-ids",
        help="Comma-separated device IDs for the fused text-encoder QPC.",
    )
    parser.add_argument(
        "--stage-device-sessions",
        action="store_true",
        help="Release text QPCs before loading the DiT QPC; supports a three-SOC staged run in text mode.",
    )
    parser.add_argument("--aic-hw-version", choices=("ai100", "ai200"))
    parser.add_argument("--aic-num-cores", type=int)
    parser.add_argument("--device-ids", help="Comma-separated QAIC device IDs, for example `0`.")
    parser.add_argument("--llada2-device-ids", help="Override device IDs for the LLaDA2 text QPC.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cpu_dtype = torch.bfloat16 if args.cpu_dtype == "bfloat16" else torch.float32
    result = pipe(
        prompt=args.prompt,
        generation_mode=args.generation_mode,
        image_path=args.input_image,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cpu_dtype=cpu_dtype,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        encoder_qpc_path=args.encoder_qpc_path,
        decoder_qpc_path=args.decoder_qpc_path,
        encoder_compile_dir=args.encoder_compile_dir,
        decoder_compile_dir=args.decoder_compile_dir,
        encoder_export_dir=args.encoder_export_dir,
        decoder_export_dir=args.decoder_export_dir,
        dit_qpc_path=args.dit_qpc_path,
        composite_text_encoder_qpc_path=args.composite_text_encoder_qpc_path,
        queryformer_qpc_path=args.queryformer_qpc_path,
        text_projector_qpc_path=args.text_projector_qpc_path,
        llada2_text_qpc_path=args.llada2_text_qpc_path,
        llada2_vq_qpc_path=args.llada2_vq_qpc_path,
        sigvq_tokens_qpc_path=args.sigvq_tokens_qpc_path,
        dit_compile_dir=args.dit_compile_dir,
        dit_export_dir=args.dit_export_dir,
        dit_layer_count=args.dit_layer_count,
        dit_use_onnx_subfunctions=args.dit_use_onnx_subfunctions,
        dit_device_ids=args.dit_device_ids,
        composite_text_encoder_device_ids=args.composite_text_encoder_device_ids,
        stage_device_sessions=args.stage_device_sessions,
        aic_hw_version=args.aic_hw_version,
        aic_num_cores=args.aic_num_cores,
        device_ids=args.device_ids,
        llada2_device_ids=args.llada2_device_ids,
        output_path=args.output,
        log_path=args.log,
        max_sequence_length=args.max_sequence_length,
    )
    for name, elapsed in result.timings.items():
        logger.info("TIMING %s | %.2f seconds", name, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
