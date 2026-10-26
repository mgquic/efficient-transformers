# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Opt-in reduced-depth model surgery for LLaDA-Image export experiments.

This module is intentionally separate from the normal model wrappers.  The
reduced model is useful for export/compile and latency smoke tests only; it is
not expected to preserve image quality.  The caller must apply the surgery
after loading the checkpoint and before configuring/exporting QEff modules.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from torch import nn


def _replace_with_first_last(
    parent: Any,
    attribute: str,
    report: MutableMapping[str, dict[str, int]],
    *,
    preserve_first: bool = False,
) -> None:
    """Keep the first and last modules in a repeated ``ModuleList``.

    ``preserve_first`` is used for ``UNetMidBlock2D`` residuals, whose first
    residual is structural and is followed by the repeated attention/residual
    pairs.  In that case the first structural residual plus the first and last
    repeated residuals are retained.
    """

    modules = getattr(parent, attribute, None)
    if not isinstance(modules, nn.ModuleList):
        return

    original_length = len(modules)
    if preserve_first:
        if original_length <= 3:
            return
        retained = [modules[0], modules[1], modules[-1]]
    else:
        if original_length <= 2:
            return
        retained = [modules[0], modules[-1]]

    setattr(parent, attribute, nn.ModuleList(retained))
    report[attribute] = {"before": original_length, "after": len(retained)}


def _trim_queryformer(queryformer: Any, report: MutableMapping[str, Any]) -> None:
    _replace_with_first_last(queryformer, "query_blocks", report["queryformer"])


def _trim_llada2(text_encoder: Any, report: MutableMapping[str, Any]) -> None:
    from QEfficient.transformers.models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelAdapter

    language_model = LLaDA2MoeModelAdapter.language_model(text_encoder)
    _replace_with_first_last(language_model, "layers", report["llada2_moe"])


def _trim_text_projection(text_projection: Any, report: MutableMapping[str, Any]) -> None:
    _replace_with_first_last(text_projection, "layers", report["text_projection"])


def _trim_dit(transformer: Any, report: MutableMapping[str, Any]) -> None:
    dit_report = report["dit"]
    for attribute in ("noise_refiner", "context_refiner", "sigvq_refiner", "layers"):
        _replace_with_first_last(transformer, attribute, dit_report)


def _trim_vae_decoder(vae: Any, report: MutableMapping[str, Any]) -> None:
    """Reduce repeated decoder blocks while preserving resolution changes.

    All VAE decoder ``up_blocks`` and their upsamplers remain present.  Within
    each up block, only the first and last residual/attention blocks are kept.
    The mid block retains its structural first residual and the first/last
    repeated residual-attention pairs.
    """

    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        raise ValueError("The LLaDA-Image VAE does not expose a decoder module")

    decoder_report = report["vae_decoder"]
    mid_block = getattr(decoder, "mid_block", None)
    if mid_block is not None:
        mid_report: dict[str, dict[str, int]] = {}
        _replace_with_first_last(mid_block, "resnets", mid_report, preserve_first=True)
        _replace_with_first_last(mid_block, "attentions", mid_report)
        if mid_report:
            decoder_report["mid_block"] = mid_report

    up_blocks = getattr(decoder, "up_blocks", None)
    if not isinstance(up_blocks, nn.ModuleList):
        return

    for index, up_block in enumerate(up_blocks):
        block_report: dict[str, dict[str, int]] = {}
        for attribute in ("resnets", "attentions", "transformer_blocks"):
            _replace_with_first_last(up_block, attribute, block_report)
        if block_report:
            decoder_report[f"up_blocks.{index}"] = block_report


def truncate_llada_image_pipeline(pipeline: Any) -> dict[str, Any]:
    """Retain first/last repeated layers in a loaded LLaDA-Image pipeline.

    The function mutates the pipeline in place and returns a report.  It
    accepts either ``QEffLLaDAImagePipeline`` or the underlying reference
    pipeline, which keeps it useful for both fused and non-fused export tests.
    """

    model = getattr(pipeline, "model", pipeline)
    report: dict[str, Any] = {
        "queryformer": {},
        "llada2_moe": {},
        "text_projection": {},
        "dit": {},
        "vae_decoder": {},
    }

    _trim_queryformer(model.queryformer, report)
    _trim_llada2(model.text_encoder, report)
    _trim_text_projection(model.text_projection, report)
    _trim_dit(model.transformer, report)
    _trim_vae_decoder(model.vae, report)
    return report
