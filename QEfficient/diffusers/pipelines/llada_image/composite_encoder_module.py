# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Core fused text-encoder component for LLaDA-Image."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import CustomOpsTransform
from QEfficient.diffusers.models.transformers.transformer_llada_image import (
    LLaDAImageQueryFormerModel,
    LLaDAImageTextProjectionModel,
)
from QEfficient.transformers.models.llada2_moe import modeling_llada2_moe as llada2


PROMPT_LENGTH = 256
CAPTION_LENGTH = 512


class LLaDAImageCompositeTextEncoderGraph(nn.Module):
    """Fixed-shape QueryFormer -> LLaDA2 -> text-projection graph."""

    def __init__(self, queryformer: nn.Module, llada2_graph: nn.Module, text_projection: nn.Module):
        super().__init__()
        self.queryformer = queryformer
        self.llada2 = llada2_graph
        self.text_projection = text_projection
        self.config = llada2_graph.config
        self.num_queries = int(queryformer.config.num_queries)
        if self.num_queries + PROMPT_LENGTH != CAPTION_LENGTH:
            raise ValueError(
                f"Expected {PROMPT_LENGTH} prompt tokens plus {self.num_queries} queries to equal "
                f"{CAPTION_LENGTH}, but they do not."
            )
        self.register_buffer("positions", torch.arange(CAPTION_LENGTH, dtype=torch.long), persistent=False)

    def _pack_caption_inputs(
        self,
        inputs_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        query_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, prompt_length, hidden_size = inputs_embeds.shape
        if prompt_length != PROMPT_LENGTH:
            raise ValueError(f"Expected {PROMPT_LENGTH} prompt slots, got {prompt_length}")
        if tuple(prompt_attention_mask.shape) != (batch_size, PROMPT_LENGTH):
            raise ValueError(
                f"Expected prompt attention mask {(batch_size, PROMPT_LENGTH)}, "
                f"got {tuple(prompt_attention_mask.shape)}"
            )
        if tuple(query_embeds.shape[:2]) != (batch_size, self.num_queries):
            raise ValueError(
                f"Expected QueryFormer output {(batch_size, self.num_queries, hidden_size)}, "
                f"got {tuple(query_embeds.shape)}"
            )

        prompt_mask = prompt_attention_mask.to(dtype=torch.bool)
        text_length = prompt_mask.to(dtype=torch.long).sum(dim=1, keepdim=True)
        positions = self.positions.to(device=inputs_embeds.device).expand(batch_size, -1)
        is_text = positions < text_length
        is_query = (positions >= text_length) & (positions < text_length + self.num_queries)
        is_valid = is_text | is_query

        text_indices = positions.clamp(max=PROMPT_LENGTH - 1)
        query_indices = (positions - text_length).clamp(min=0, max=self.num_queries - 1)
        text_values = torch.gather(
            inputs_embeds,
            1,
            text_indices.unsqueeze(-1).expand(batch_size, CAPTION_LENGTH, hidden_size),
        )
        query_values = torch.gather(
            query_embeds,
            1,
            query_indices.unsqueeze(-1).expand(batch_size, CAPTION_LENGTH, hidden_size),
        )
        packed_embeds = torch.where(is_text.unsqueeze(-1), text_values, query_values)
        packed_embeds = packed_embeds * is_valid.unsqueeze(-1).to(dtype=packed_embeds.dtype)

        row_positions = positions.unsqueeze(-1)
        column_positions = positions.unsqueeze(-2)
        combined_length = text_length + self.num_queries
        valid_domain = (row_positions < combined_length.unsqueeze(-1)) & (
            column_positions < combined_length.unsqueeze(-1)
        )
        padded_domain = (row_positions >= combined_length.unsqueeze(-1)) & (
            column_positions >= combined_length.unsqueeze(-1)
        )
        text_rows = row_positions < text_length.unsqueeze(-1)
        query_columns = (column_positions >= text_length.unsqueeze(-1)) & (
            column_positions < combined_length.unsqueeze(-1)
        )
        allowed = (valid_domain & ~(text_rows & query_columns)) | padded_domain
        mask_value = torch.finfo(inputs_embeds.dtype).min
        attention_mask = torch.where(
            allowed,
            torch.zeros((), device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            torch.full((), mask_value, device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        ).unsqueeze(1)
        return packed_embeds, attention_mask, positions, is_valid

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        query_embeds = self.queryformer(inputs_embeds, attention_mask, return_dict=False)[0]
        packed_embeds, llada2_attention_mask, position_ids, projected_attention_mask = self._pack_caption_inputs(
            inputs_embeds, attention_mask, query_embeds
        )
        hidden_states = self.llada2(packed_embeds, llada2_attention_mask, position_ids)
        return self.text_projection(
            hidden_states,
            attention_mask=projected_attention_mask,
            return_dict=False,
        )[0]


class QEffLLaDAImageCompositeTextEncoder(QEFFBaseModel):
    """QEff export/compile adapter for the fused LLaDA-Image text encoder."""

    _pytorch_transforms = [CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    def __init__(
        self,
        graph: LLaDAImageCompositeTextEncoderGraph,
        *,
        llada2_implementation: str,
        reuse_pipeline_components: bool = False,
        **kwargs,
    ):
        super().__init__(graph, module_name="llada_image_composite_text_encoder", **kwargs)
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.hash_params["llada2_wrapper"] = llada2_implementation
        self.hash_params["router_fp32_npi"] = False
        self.hash_params["caption_contract"] = "queryformer_llada2_text_projection_s256_s512"
        self.reuse_pipeline_components = reuse_pipeline_components

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        llada2_implementation: str = "default",
        torch_dtype: torch.dtype = torch.float32,
        queryformer: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        text_projection: nn.Module | None = None,
    ) -> "QEffLLaDAImageCompositeTextEncoder":
        model_path = Path(model_path).expanduser().resolve()
        supplied_components = (queryformer, text_encoder, text_projection)
        if any(component is not None for component in supplied_components) and not all(
            component is not None for component in supplied_components
        ):
            raise ValueError("queryformer, text_encoder, and text_projection must be supplied together")
        reuse_pipeline_components = all(component is not None for component in supplied_components)

        if queryformer is None:
            queryformer = LLaDAImageQueryFormerModel.from_pretrained(
                model_path / "queryformer", torch_dtype=torch_dtype
            )
        if text_projection is None:
            text_projection = LLaDAImageTextProjectionModel.from_pretrained(
                model_path / "text_projection", torch_dtype=torch_dtype
            )

        if llada2_implementation != "default":
            raise ValueError(
                "Only the default LLaDA2 implementation is supported; "
                f"got {llada2_implementation!r}."
            )
        llada2._register_llada2_rope_compatibility()
        if text_encoder is None:
            text_encoder = AutoModel.from_pretrained(
                model_path / "text_encoder",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation="eager",
            )
        llada2_backbone = text_encoder
        llada2_backbone, transformed = llada2.LLaDA2MoeTransform.apply(llada2_backbone)
        if not transformed:
            raise RuntimeError("LLaDA2 transforms did not match the Turbo text encoder.")
        llada2_graph = llada2.LLaDA2EncoderGraph(llada2_backbone.eval(), output_logits=False)

        graph = LLaDAImageCompositeTextEncoderGraph(
            queryformer.eval(),
            llada2_graph,
            text_projection.eval(),
        )
        return cls(
            graph,
            llada2_implementation=llada2_implementation,
            reuse_pipeline_components=reuse_pipeline_components,
            pretrained_model_name_or_path=str(model_path / "text_encoder"),
        )

    @property
    def get_model_config(self) -> dict:
        config = self.model.config.__dict__.copy()
        if isinstance(config.get("_name_or_path"), Path):
            config["_name_or_path"] = str(config["_name_or_path"])
        return config

    def export(
        self,
        *,
        export_dir: str | Path,
        offload_pt_weights: bool = False,
        use_onnx_subfunctions: bool = False,
        **kwargs,
    ) -> Path:
        del kwargs
        if self.reuse_pipeline_components:
            # The reference pipeline still needs its embedding lookup after the
            # fused export, so do not replace the shared weights with meta tensors.
            offload_pt_weights = False
        if use_onnx_subfunctions:
            raise ValueError("The fused LLaDA-Image encoder requires use_onnx_subfunctions=False")
        example_inputs, dynamic_axes, output_names = self.get_onnx_params()
        self.hash_params.update(
            {
                "batch_size": 1,
                "prompt_length": PROMPT_LENGTH,
                "caption_length": CAPTION_LENGTH,
                "supported_resolutions": [256, 1024],
            }
        )
        return Path(
            self._export(
                example_inputs=example_inputs,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_dir=Path(export_dir),
                offload_pt_weights=offload_pt_weights,
                dynamo=False,
                use_onnx_subfunctions=False,
            )
        )

    def get_onnx_params(self) -> tuple[dict[str, torch.Tensor], dict, list[str]]:
        """Return the fixed tensor contract for the fused text encoder."""
        dtype = next(self.model.parameters()).dtype
        hidden_size = int(self.model.config.hidden_size)
        return (
            {
                "inputs_embeds": torch.randn(1, PROMPT_LENGTH, hidden_size, dtype=dtype),
                "attention_mask": torch.ones(1, PROMPT_LENGTH, dtype=torch.bool),
            },
            {},
            ["prompt_embeds"],
        )

    def compile(
        self,
        *,
        onnx_path: str | Path | None = None,
        compile_dir: str | Path,
        specializations: list[dict] | None = None,
        num_devices: int | None = None,
        mdp_ts_num_devices: int | None = None,
        aic_hw_version: str = "ai100",
        aic_num_cores: int = 16,
        mxfp6_matmul: bool = True,
        convert_to_fp16: bool = True,
        use_onnx_subfunctions: bool = False,
        **compiler_options,
    ) -> Path:
        del use_onnx_subfunctions
        device_count = int(num_devices or mdp_ts_num_devices or 2)
        if device_count not in (1, 2, 16):
            raise ValueError("The fused LLaDA-Image encoder supports one, two, or sixteen tensor slices.")
        compiler_options.pop("num_devices", None)
        compiler_options.pop("mdp_ts_num_devices", None)
        # These options are part of the adapter contract rather than free-form
        # compiler overrides.  Remove duplicates when called through the
        # generic pipeline config manager.
        compiler_options.pop("node_precision_info", None)
        compiler_options.pop("offload_pt_weights", None)
        return Path(
            self._compile(
                onnx_path=onnx_path,
                compile_dir=compile_dir,
                specializations=specializations or [{"batch_size": 1}],
                mdp_ts_num_devices=device_count,
                aic_hw_version=aic_hw_version,
                aic_num_cores=aic_num_cores,
                mxfp6_matmul=mxfp6_matmul,
                convert_to_fp16=convert_to_fp16,
                node_precision_info=False,
                use_onnx_subfunctions=False,
                offload_pt_weights=False,
                **compiler_options,
            )
        )


def validate_composite_text_encoder(adapter: QEffLLaDAImageCompositeTextEncoder) -> None:
    """Validate the fused graph against its staged calculation in PyTorch."""
    graph = adapter.model
    dtype = next(graph.parameters()).dtype
    prompt_mask = torch.zeros(1, PROMPT_LENGTH, dtype=torch.bool)
    prompt_mask[:, :37] = True
    inputs_embeds = torch.randn(1, PROMPT_LENGTH, graph.config.hidden_size, dtype=dtype)
    with torch.no_grad():
        fused = graph(inputs_embeds, prompt_mask)
        query_embeds = graph.queryformer(inputs_embeds, prompt_mask, return_dict=False)[0]
        packed, llada_mask, positions, projected_mask = graph._pack_caption_inputs(
            inputs_embeds,
            prompt_mask,
            query_embeds,
        )
        hidden_states = graph.llada2(packed, llada_mask, positions)
        staged = graph.text_projection(
            hidden_states,
            attention_mask=projected_mask,
            return_dict=False,
        )[0]
    max_abs_diff = float((fused - staged).abs().max())
    if not torch.isfinite(fused).all() or max_abs_diff > 1e-5:
        raise RuntimeError(f"Composite PyTorch validation failed: max_abs_diff={max_abs_diff}")


# Compatibility aliases for existing experiment imports and artifact metadata.
QEffTurboCompositeTextEncoder = QEffLLaDAImageCompositeTextEncoder
TurboCompositeTextEncoderGraph = LLaDAImageCompositeTextEncoderGraph


__all__ = [
    "CAPTION_LENGTH",
    "PROMPT_LENGTH",
    "LLaDAImageCompositeTextEncoderGraph",
    "QEffLLaDAImageCompositeTextEncoder",
    "QEffTurboCompositeTextEncoder",
    "TurboCompositeTextEncoderGraph",
    "validate_composite_text_encoder",
]
