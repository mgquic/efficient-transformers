# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""Fixed-shape export and QAIC compile adapters for LLaDA-Image components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.base.onnx_transforms import FP16ClipTransform, SplitTensorsTransform
from QEfficient.diffusers.models.pytorch_transforms import CustomOpsTransform, LLaDAImageComponentTransform
from QEfficient.utils import constants


class _StaticSigVQPositionEmbedding(nn.Module):
    """Precompute SigVQ's fixed-resolution bilinear position interpolation."""

    def __init__(self, embeddings: nn.Embedding, grid_height: int, grid_width: int):
        super().__init__()
        source = embeddings.weight.detach().float()
        source_size = int(source.shape[0] ** 0.5)
        source = source.reshape(source_size, source_size, source.shape[-1]).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(source, size=(grid_height, grid_width), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0).flatten(1).transpose(0, 1).unsqueeze(0).to(dtype=embeddings.weight.dtype)
        self.register_buffer("position_embedding", resized, persistent=True)

    def forward(self, hidden_states: torch.Tensor, _grid_height: int, _grid_width: int) -> torch.Tensor:
        return hidden_states + self.position_embedding.to(dtype=hidden_states.dtype).expand(hidden_states.shape[0], -1, -1)


class _LLaDAImageComponentGraph(nn.Module):
    """Tensor-only boundary around one LLaDA-Image component."""

    def __init__(self, model: nn.Module, component: str):
        super().__init__()
        self.model = model
        self.config = model.config
        self.component = component

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        token_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ):
        if self.component == "queryformer":
            return self.model(inputs_embeds, attention_mask, return_dict=False)[0]
        if self.component == "text_projector":
            return self.model(hidden_states, attention_mask=attention_mask, return_dict=False)[0]
        if self.component == "sigvq_tokens":
            return self.model(token_ids=token_ids, return_dict=False)
        if self.component == "sigvq_image":
            return self.model(pixel_values=pixel_values, return_dict=False)
        raise ValueError(f"Unsupported LLaDA-Image component: {self.component}")


class QEffLLaDAImageComponentGraph(_LLaDAImageComponentGraph):
    """QEff export graph around one transformed LLaDA-Image component."""


class QEffLLaDAImageComponent(QEFFBaseModel):
    """Export/compile adapter for a fixed-shape LLaDA-Image component graph."""

    _pytorch_transforms = [LLaDAImageComponentTransform, CustomOpsTransform]
    _onnx_transforms = [FP16ClipTransform, SplitTensorsTransform]

    _COMPONENTS = {"queryformer", "text_projector", "sigvq_tokens", "sigvq_image"}

    def __init__(self, model: nn.Module, *, component: str, **kwargs: Any):
        if component not in self._COMPONENTS:
            raise ValueError(f"Unsupported LLaDA-Image component: {component}")
        if component == "sigvq_image":
            model.visual.embeddings = _StaticSigVQPositionEmbedding(
                model.visual.embeddings.position_embedding,
                grid_height=kwargs.pop("image_height", 256) // int(model.config.patch_size),
                grid_width=kwargs.pop("image_width", 256) // int(model.config.patch_size),
            )
        graph = QEffLLaDAImageComponentGraph(model, component)
        super().__init__(graph, module_name=f"llada_image_{component}", **kwargs)
        self.component = component
        self.hash_params["llada_image_component"] = component

    @property
    def get_model_config(self):
        config = self.model.config.__dict__.copy()
        name_or_path = config.get("_name_or_path")
        if isinstance(name_or_path, Path):
            config["_name_or_path"] = str(name_or_path)
        return config

    def _example_inputs(self, *, batch_size: int, sequence_length: int, image_height: int, image_width: int):
        dtype = next(self.model.parameters()).dtype
        if self.component == "queryformer":
            inputs = {
                "inputs_embeds": torch.randn(batch_size, sequence_length, self.config.hidden_size, dtype=dtype),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.bool),
            }
            return inputs, {}, ["query_embeds"]
        if self.component == "text_projector":
            inputs = {
                "hidden_states": torch.randn(batch_size, sequence_length, self.config.hidden_size, dtype=dtype),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.bool),
            }
            return inputs, {}, ["projected_hidden_states"]
        if self.component == "sigvq_tokens":
            inputs = {"token_ids": torch.zeros(batch_size, sequence_length, dtype=torch.int64)}
            return inputs, {}, ["semantic_features", "token_ids_out"]
        inputs = {"pixel_values": torch.randn(batch_size, 3, image_height, image_width, dtype=dtype)}
        return inputs, {}, ["semantic_features", "token_ids_out"]

    def export(
        self,
        export_dir: str | Path | None = None,
        *,
        batch_size: int = 1,
        sequence_length: int = 512,
        image_height: int = 256,
        image_width: int = 256,
        offload_pt_weights: bool = True,
        **kwargs: Any,
    ) -> str:
        example_inputs, dynamic_axes, output_names = self._example_inputs(
            batch_size=batch_size,
            sequence_length=sequence_length,
            image_height=image_height,
            image_width=image_width,
        )
        self.hash_params.update(
            {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "image_height": image_height,
                "image_width": image_width,
            }
        )
        return self._export(
            example_inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_dir=Path(export_dir or "."),
            offload_pt_weights=offload_pt_weights,
            dynamo=False,
            **kwargs,
        )

    def compile(
        self,
        *,
        onnx_path: str | Path | None = None,
        compile_dir: str | Path,
        export_dir: str | Path | None = None,
        batch_size: int = 1,
        sequence_length: int = 512,
        image_height: int = 256,
        image_width: int = 256,
        aic_hw_version: str | None = None,
        aic_num_cores: int | None = None,
        specializations: list[dict[str, Any]] | None = None,
        **compiler_options: Any,
    ) -> str:
        compile_dir = Path(compile_dir)
        if onnx_path is None:
            onnx_path = self.export(
                export_dir=export_dir or compile_dir,
                batch_size=batch_size,
                sequence_length=sequence_length,
                image_height=image_height,
                image_width=image_width,
                offload_pt_weights=False,
            )
        compiler_options.setdefault("aic_hw_version", aic_hw_version or constants.DEFAULT_AIC_HW_VERSION)
        compiler_options.setdefault("aic_num_cores", aic_num_cores or constants.DEFAULT_AIC_NUM_CORES)
        compiler_options.setdefault("convert_to_fp16", True)
        if specializations is not None:
            self.qpc_path = self._compile(
                onnx_path=onnx_path,
                compile_dir=compile_dir,
                specializations=specializations,
                offload_pt_weights=False,
                **compiler_options,
            )
            return self.qpc_path

        self.qpc_path = self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            specializations=[{"batch_size": batch_size}],
            offload_pt_weights=False,
            **compiler_options,
        )
        return self.qpc_path
