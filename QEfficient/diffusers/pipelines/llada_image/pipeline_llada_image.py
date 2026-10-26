# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKLFlux2
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from QEfficient.diffusers.models.transformers.transformer_llada_image import (
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQModel,
    LLaDAImageTextProjectionModel,
    LLaDAImageTransformer2DModel,
)
from QEfficient.diffusers.pipelines.llada_image.pipeline_output import LLaDAImagePipelineOutput
from QEfficient.transformers.llada2_utils import restrict_vq_logits


logger = logging.get_logger(__name__)


@contextmanager
def _component_timing(timings: dict[str, float] | None, name: str):
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


class LLaDAImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for LLaDA-Image text-to-image generation, VQ-conditioned generation, and single-image editing.

    Args:
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used for denoising.
        vae ([`AutoencoderKLFlux2`]):
            Flux2 VAE used to encode reference images and decode generated latents.
        text_encoder (`transformers.PreTrainedModel`):
            LLaDA2 conditional-generation model. It must expose `get_input_embeddings()` and its language backbone as
            `model`.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            Tokenizer paired with the LLaDA2 text encoder.
        queryformer ([`LLaDAImageQueryFormerModel`]):
            QueryFormer that refines the learnable generation queries.
        text_projection ([`LLaDAImageTextProjectionModel`]):
            Connector and projector that map LLaDA2 hidden states to denoiser caption features.
        sigvq ([`LLaDAImageSigVQModel`]):
            GLM SigVQ component that embeds MLLM-generated VQ tokens and encodes editing reference images.
        transformer ([`LLaDAImageTransformer2DModel`]):
            Denoising transformer.
    """

    model_cpu_offload_seq = "text_encoder->queryformer->text_projection->sigvq->transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        queryformer: LLaDAImageQueryFormerModel,
        text_projection: LLaDAImageTextProjectionModel,
        sigvq: LLaDAImageSigVQModel,
        transformer: LLaDAImageTransformer2DModel,
        queryformer_session=None,
        composite_text_encoder_session=None,
        text_projection_session=None,
        sigvq_tokens_session=None,
        llada2_text_session=None,
        llada2_vq_session=None,
        dit_session=None,
        vae_encoder_session=None,
        vae_decoder_session=None,
    ):
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            queryformer=queryformer,
            text_projection=text_projection,
            sigvq=sigvq,
            transformer=transformer,
        )
        # These are runtime-only adapters, not Diffusers modules. They are set
        # by the AIC runner after the CPU pipeline is loaded.
        self.queryformer_session = queryformer_session
        self.composite_text_encoder_session = composite_text_encoder_session
        self.text_projection_session = text_projection_session
        self.sigvq_tokens_session = sigvq_tokens_session
        self.llada2_text_session = llada2_text_session
        self.llada2_vq_session = llada2_vq_session
        self.dit_session = dit_session
        self.vae_encoder_session = vae_encoder_session
        self.vae_decoder_session = vae_decoder_session
        self.component_timings: dict[str, float] | None = None

        # These buffers are needed to denormalize latents even after the VAE
        # weights are offloaded to meta following export.
        self._vae_latent_mean = self.vae.bn.running_mean.detach().cpu().clone()
        self._vae_latent_var = self.vae.bn.running_var.detach().cpu().clone()

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if self.vae is not None else 8
        self.latent_scale_factor = self.vae_scale_factor * 2
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        torch_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "LLaDAImagePipeline":
        """Load all LLaDA-Image components from a converted model directory or Hugging Face repository.

        This model stores a LLaDA2 text encoder that requires `trust_remote_code=True`, so its components are loaded
        explicitly instead of relying on the generic Diffusers pipeline resolver.
        """
        model_path = Path(pretrained_model_name_or_path)
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download

            model_path = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                )
            )

        if not (model_path / "model_index.json").is_file():
            raise ValueError(
                "Expected a converted LLaDA-Image model directory containing `model_index.json`, got "
                f"{model_path}."
            )

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path / "scheduler")
        vae = AutoencoderKLFlux2.from_pretrained(model_path / "vae", torch_dtype=torch_dtype)
        from QEfficient.transformers.models.llada2_moe.modeling_llada2_moe import (
            LLaDA2MoeModelAdapter,
            _register_llada2_rope_compatibility,
        )

        _register_llada2_rope_compatibility()
        text_encoder_kwargs = {"dtype": torch_dtype, "trust_remote_code": True}
        if device is not None:
            text_encoder_kwargs["device_map"] = {"": device}
        text_encoder = AutoModel.from_pretrained(model_path / "text_encoder", **text_encoder_kwargs)
        LLaDA2MoeModelAdapter.install_cached_rope(text_encoder)
        tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
        queryformer = LLaDAImageQueryFormerModel.from_pretrained(model_path / "queryformer", torch_dtype=torch_dtype)
        text_projection = LLaDAImageTextProjectionModel.from_pretrained(
            model_path / "text_projection", torch_dtype=torch_dtype
        )
        sigvq = LLaDAImageSigVQModel.from_pretrained(model_path / "sigvq", torch_dtype=torch_dtype)
        transformer = LLaDAImageTransformer2DModel.from_pretrained(model_path / "transformer", torch_dtype=torch_dtype)

        if device is not None:
            vae = vae.to(device)
            queryformer = queryformer.to(device)
            text_projection = text_projection.to(device)
            sigvq = sigvq.to(device)
            transformer = transformer.to(device)

        pipeline = cls(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            queryformer=queryformer,
            text_projection=text_projection,
            sigvq=sigvq,
            transformer=transformer,
        )
        # Keep the resolved local directory available to the QEff wrapper so a
        # fused export adapter can be built without resolving the model again.
        pipeline._llada_image_model_path = model_path
        return pipeline

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels // 4, height * 2, width * 2)

    def _encode_text(
        self,
        prompts: list[str],
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with _component_timing(self.component_timings, "component.text.prompt_formatting"):
            formatted_prompts = [
                "<role>HUMAN</role> Generate an image.\n<role>ASSISTANT</role>\n<IMAGE1>"
                if prompt is None
                else f"<role>HUMAN</role> Generate an image: {prompt.strip()}\n<role>ASSISTANT</role>\n<IMAGE1>"
                for prompt in prompts
            ]
        with _component_timing(self.component_timings, "component.text.tokenization"):
            text_inputs = self.tokenizer(
                formatted_prompts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_sequence_length,
                return_tensors="pt",
            )
        text_encoder_device = self.text_encoder.device
        with _component_timing(self.component_timings, "component.text.input_transfer_and_mask"):
            input_ids = text_inputs.input_ids.to(text_encoder_device)
            attention_mask = text_inputs.attention_mask.to(text_encoder_device).bool()
        with _component_timing(self.component_timings, "component.text.embedding_lookup"):
            inputs_embeds = self.text_encoder.get_input_embeddings()(input_ids)

        if self.composite_text_encoder_session is not None:
            session = self.composite_text_encoder_session
            if inputs_embeds.shape[0] != 1:
                raise ValueError("The fused text encoder AIC graph supports batch size one")
            if inputs_embeds.shape[1] > session.input_length:
                raise ValueError(
                    f"Prompt tokens ({inputs_embeds.shape[1]}) exceed the fused text encoder graph length "
                    f"({session.input_length})."
                )
            with _component_timing(self.component_timings, "component.fused_text_encoder.input_padding"):
                padded_embeds = inputs_embeds.new_zeros(
                    (1, session.input_length, inputs_embeds.shape[-1])
                )
                padded_embeds[:, : inputs_embeds.shape[1]] = inputs_embeds
                padded_mask = attention_mask.new_zeros((1, session.input_length))
                padded_mask[:, : attention_mask.shape[1]] = attention_mask
            with _component_timing(self.component_timings, "component.fused_text_encoder"):
                with _component_timing(self.component_timings, "component.fused_text_encoder.run"):
                    prompt_embeds = session.run(padded_embeds, padded_mask).to(device=text_encoder_device)
            with _component_timing(self.component_timings, "component.fused_text_encoder.output_postprocess"):
                text_length = attention_mask.to(dtype=torch.long).sum(dim=1, keepdim=True)
                projected_attention_mask = torch.zeros(
                    (1, session.output_length), dtype=attention_mask.dtype, device=text_encoder_device
                )
                valid_length = text_length + session.num_queries
                positions = torch.arange(session.output_length, device=text_encoder_device).unsqueeze(0)
                projected_attention_mask = positions < valid_length
            if getattr(self, "stage_device_sessions", False):
                with _component_timing(self.component_timings, "component.fused_text_encoder.teardown"):
                    session.close()
                self.composite_text_encoder_session = None
            return prompt_embeds, projected_attention_mask

        if self.queryformer_session is None:
            with _component_timing(self.component_timings, "component.queryformer"):
                query_embeds = self.queryformer(
                    inputs_embeds.to(device=self.queryformer.device, dtype=self.queryformer.dtype),
                    attention_mask.to(self.queryformer.device),
                ).query_embeds.to(device=text_encoder_device, dtype=inputs_embeds.dtype)
        else:
            session = self.queryformer_session
            if inputs_embeds.shape[0] != 1:
                raise ValueError("The initial QueryFormer AIC graph supports batch size one")
            if inputs_embeds.shape[1] > session.input_length:
                raise ValueError(
                    f"Prompt tokens ({inputs_embeds.shape[1]}) exceed the QueryFormer graph length "
                    f"({session.input_length})."
                )
            queryformer_embeds = inputs_embeds.new_zeros(
                (1, session.input_length, inputs_embeds.shape[-1])
            )
            queryformer_embeds[:, : inputs_embeds.shape[1]] = inputs_embeds
            queryformer_mask = attention_mask.new_zeros((1, session.input_length))
            queryformer_mask[:, : attention_mask.shape[1]] = attention_mask
            with _component_timing(self.component_timings, "component.queryformer"):
                query_embeds = session.run(queryformer_embeds, queryformer_mask).to(
                    device=text_encoder_device, dtype=inputs_embeds.dtype
                )
            if getattr(self, "stage_device_sessions", False):
                session.close()
                self.queryformer_session = None
        text_length = inputs_embeds.shape[1]
        inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(attention_mask.shape[0], query_embeds.shape[1])],
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(position_ids < 0, 0)

        mask_value = torch.finfo(inputs_embeds.dtype).min
        backbone_attention_mask = attention_mask[:, None, None, :].expand(-1, 1, attention_mask.shape[1], -1)
        backbone_attention_mask = torch.where(
            backbone_attention_mask,
            torch.zeros((), dtype=inputs_embeds.dtype, device=text_encoder_device),
            torch.full((), mask_value, dtype=inputs_embeds.dtype, device=text_encoder_device),
        )
        backbone_attention_mask[:, :, :text_length, text_length:] = mask_value
        if self.llada2_text_session is None:
            with _component_timing(self.component_timings, "component.llada2_text"):
                hidden_states = self.text_encoder.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=backbone_attention_mask,
                    position_ids=position_ids,
                    return_dict=True,
                ).last_hidden_state
            projected_attention_mask = attention_mask
        else:
            session = self.llada2_text_session
            if inputs_embeds.shape[0] != 1:
                raise ValueError("The initial LLaDA2 AIC text graph supports batch size one")
            sequence_length = session.sequence_length
            if inputs_embeds.shape[1] > sequence_length:
                raise ValueError(
                    f"Prompt plus QueryFormer tokens ({inputs_embeds.shape[1]}) exceed the LLaDA2 text graph length "
                    f"({sequence_length})."
                )
            padded_embeds = inputs_embeds.new_zeros((1, sequence_length, inputs_embeds.shape[-1]))
            padded_embeds[:, : inputs_embeds.shape[1]] = inputs_embeds
            padded_mask = torch.full(
                (1, 1, sequence_length, sequence_length),
                torch.finfo(inputs_embeds.dtype).min,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            padded_mask[:, :, : inputs_embeds.shape[1], : inputs_embeds.shape[1]] = backbone_attention_mask
            # Give padded query rows a finite self-attention domain. Their
            # outputs are masked out before DiT, but an all--inf softmax would
            # otherwise create NaNs that could leak through the projector.
            padded_mask[:, :, inputs_embeds.shape[1] :, inputs_embeds.shape[1] :] = 0
            padded_positions = torch.arange(sequence_length, device=inputs_embeds.device).unsqueeze(0)
            with _component_timing(self.component_timings, "component.llada2_text"):
                hidden_states = session.text(padded_embeds, padded_mask, padded_positions)
            if getattr(self, "stage_device_sessions", False):
                session.close()
                self.llada2_text_session = None
            projected_attention_mask = attention_mask.new_zeros((1, sequence_length))
            projected_attention_mask[:, : attention_mask.shape[1]] = attention_mask
        if self.text_projection_session is None:
            with _component_timing(self.component_timings, "component.text_projection"):
                prompt_embeds = self.text_projection(
                    hidden_states.to(device=self.text_projection.device, dtype=self.text_projection.dtype),
                    attention_mask=projected_attention_mask.to(self.text_projection.device),
                ).hidden_states
        else:
            session = self.text_projection_session
            if hidden_states.shape[0] != 1:
                raise ValueError("The initial text projector AIC graph supports batch size one")
            if hidden_states.shape[1] > session.sequence_length:
                raise ValueError(
                    f"LLaDA2 hidden states ({hidden_states.shape[1]}) exceed the text projector graph length "
                    f"({session.sequence_length})."
                )
            projector_hidden_states = hidden_states.new_zeros(
                (1, session.sequence_length, hidden_states.shape[-1])
            )
            projector_hidden_states[:, : hidden_states.shape[1]] = hidden_states
            if projected_attention_mask.shape[1] < session.sequence_length:
                padded_attention_mask = projected_attention_mask.new_zeros((1, session.sequence_length))
                padded_attention_mask[:, : projected_attention_mask.shape[1]] = projected_attention_mask
                projected_attention_mask = padded_attention_mask
            with _component_timing(self.component_timings, "component.text_projection"):
                prompt_embeds = session.run(projector_hidden_states, projected_attention_mask)
            if getattr(self, "stage_device_sessions", False):
                session.close()
                self.text_projection_session = None
        return prompt_embeds, projected_attention_mask.to(prompt_embeds.device)

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        device = device or self._execution_device

        if prompt_embeds is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt_embeds, prompt_attention_mask = self._encode_text(prompt, max_sequence_length)
        else:
            prompt_embeds = prompt_embeds.to(device)
            prompt_attention_mask = prompt_attention_mask.to(device).bool()

        batch_size = prompt_embeds.shape[0]
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = [None] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_attention_mask = self._encode_text(
                negative_prompt, max_sequence_length
            )
        elif do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(device)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device).bool()

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                num_images_per_prompt, dim=0
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    @staticmethod
    def _pad_condition_features(
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length host features into a fixed AIC graph contract."""
        batch_size, _, feature_dim = features.shape
        padded = features.new_zeros((batch_size, target_length, feature_dim))
        padded_mask = torch.zeros((batch_size, target_length), dtype=torch.int64, device=features.device)
        for batch_index in range(batch_size):
            valid = features[batch_index][attention_mask[batch_index].bool()]
            valid_length = min(valid.shape[0], target_length)
            padded[batch_index, :valid_length] = valid[:valid_length]
            padded_mask[batch_index, :valid_length] = 1
        return padded, padded_mask

    def generate_vq_tokens(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
    ) -> torch.Tensor:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        image_token_offset = int(getattr(self.text_encoder.config, "image_token_offset", 157184))
        codebook_size = int(getattr(self.sigvq.config, "codebook_size", 16384))
        frontend_scale = max(max(height, width) / 512, 1.0)
        frontend_height = int(height / frontend_scale)
        frontend_width = int(width / frontend_scale)
        vq_height = frontend_height // 16
        vq_width = frontend_width // 16
        image_token_count = vq_height * vq_width
        system_prompt = "You are a text-to-image generation assistant."
        generated_tokens = []

        if self.llada2_vq_session is not None and len(prompts) != 1:
            raise ValueError("The initial LLaDA2 AIC VQ graph supports one prompt at a time")

        for prompt in prompts:
            text_prompt = f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role>{prompt}<role>ASSISTANT</role>"
            text_ids = self.tokenizer(text_prompt).input_ids
            image_info_ids = self.tokenizer(
                f"<|image|><|reserved_token_{vq_height}|><|reserved_token_{vq_width}|><boi><|/image|>"
            ).input_ids
            input_ids = text_ids + image_info_ids[:-1]

            uncond_prompt = (
                f"<role>SYSTEM</role> {system_prompt} <role>HUMAN</role><uncondition><role>ASSISTANT</role>"
            )
            uncond_ids = self.tokenizer(uncond_prompt).input_ids + image_info_ids[:-1]
            if self.llada2_vq_session is None:
                lm_head = self.text_encoder.model.lm_head
                original_lm_head_forward = lm_head.forward

                def masked_lm_head_forward(hidden_states):
                    logits = original_lm_head_forward(hidden_states)
                    return restrict_vq_logits(logits, image_token_offset, codebook_size)

                lm_head.forward = masked_lm_head_forward
                try:
                    with _component_timing(self.component_timings, "component.llada2_vq"):
                        output_ids = self.text_encoder.generate_bd_image_logic(
                            data={
                                "input_ids": torch.tensor(input_ids, device=self.text_encoder.device).unsqueeze(0),
                                "uncond_ids": uncond_ids,
                            },
                            block_length=32,
                            steps=8,
                            gen_length=image_token_count,
                            cfg_scale=2.0,
                        )
                finally:
                    lm_head.forward = original_lm_head_forward
            else:
                embedding_lookup = self.text_encoder.get_input_embeddings()
                with _component_timing(self.component_timings, "component.llada2_vq"):
                    output_ids = self.llada2_vq_session.generate_vq(
                        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                        torch.tensor(uncond_ids, dtype=torch.long).unsqueeze(0),
                        embedding_lookup,
                        block_length=32,
                        steps=8,
                        generation_length=image_token_count,
                        cfg_scale=2.0,
                        image_token_offset=image_token_offset,
                        codebook_size=codebook_size,
                    )
            token_ids = output_ids[0, len(input_ids) : len(input_ids) + image_token_count] - image_token_offset
            if len(token_ids) != image_token_count:
                raise ValueError(f"The MLLM generated {len(token_ids)} VQ tokens, expected {image_token_count}.")
            invalid = (token_ids < 0) | (token_ids >= self.sigvq.config.codebook_size)
            if torch.any(invalid):
                raw_token_ids = output_ids[0, len(input_ids) : len(input_ids) + image_token_count].detach().cpu()
                shifted_token_ids = token_ids.detach().cpu()
                invalid_raw = raw_token_ids[invalid.detach().cpu()].tolist()
                invalid_shifted = shifted_token_ids[invalid.detach().cpu()].tolist()
                raise ValueError(
                    "The MLLM generated token IDs outside the SigVQ codebook: "
                    f"raw_range=[{int(raw_token_ids.min())}, {int(raw_token_ids.max())}], "
                    f"shifted_range=[{int(shifted_token_ids.min())}, {int(shifted_token_ids.max())}], "
                    f"invalid_raw={invalid_raw[:32]}, invalid_shifted={invalid_shifted[:32]}"
                )
            generated_tokens.append(token_ids)

        return torch.stack(generated_tokens)

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        image: PipelineImageInput | None,
        generation_mode: str,
        height: int,
        width: int,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        callback_on_step_end_tensor_inputs: list[str],
        num_inference_steps: int,
    ) -> None:
        if generation_mode not in {"text", "vq", "editing"}:
            raise ValueError("`generation_mode` must be one of 'text', 'vq', or 'editing'.")
        if generation_mode in {"text", "vq"} and image is not None:
            raise ValueError(f"`image` must be omitted when `generation_mode='{generation_mode}'`.")
        if generation_mode == "vq" and prompt is None:
            raise ValueError("`prompt` is required when `generation_mode='vq'`.")
        if generation_mode == "editing" and image is None:
            raise ValueError("`image` is required when `generation_mode='editing'`.")
        if generation_mode == "vq" and (height % 16 != 0 or width % 16 != 0):
            raise ValueError("`height` and `width` must be divisible by 16 in VQ mode.")

        required_multiple = self.latent_scale_factor * (2 if generation_mode == "editing" else 1)
        if height <= 0 or width <= 0 or height % required_multiple != 0 or width % required_multiple != 0:
            raise ValueError(f"`height` and `width` must be divisible by {required_multiple}.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be at least 1.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Provide only one of `prompt` or `prompt_embeds`.")
        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("`prompt_attention_mask` is required with `prompt_embeds`.")
        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("`negative_prompt_attention_mask` is required with `negative_prompt_embeds`.")
        if num_images_per_prompt < 1:
            raise ValueError("`num_images_per_prompt` must be at least 1.")
        if not all(name in self._callback_tensor_inputs for name in callback_on_step_end_tensor_inputs):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` must be chosen from {self._callback_tensor_inputs}."
            )

    def _encode_source_image(
        self,
        image: PipelineImageInput,
        height: int,
        width: int,
        batch_size: int,
        num_images_per_prompt: int,
        precomputed_source_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.image_processor.preprocess(image, height=height, width=width)
        if image.shape[0] == 1 and batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1)
        if image.shape[0] != batch_size:
            raise ValueError(f"The image batch size must be 1 or {batch_size}, but is {image.shape[0]}.")
        image = image.repeat_interleave(num_images_per_prompt, dim=0)

        sigvq_pixel_values = F.interpolate(
            image.float(),
            size=(height // 2, width // 2),
            mode="bilinear",
            align_corners=False,
        )
        semantic_features = self.sigvq(
            sigvq_pixel_values.to(device=self.sigvq.device, dtype=self.sigvq.dtype)
        ).semantic_features

        if precomputed_source_latents is None:
            if self.vae_encoder_session is None:
                source_latents = self.vae.encode(
                    image.to(device=self.vae.device, dtype=self.vae.dtype)
                ).latent_dist.mode()
            else:
                source_latents = self.vae_encoder_session.run(image)
            source_latents = self._patchify_latents(source_latents)
            latent_mean = self._vae_latent_mean.view(1, -1, 1, 1).to(source_latents)
            latent_std = torch.sqrt(
                self._vae_latent_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(source_latents)
            source_latents = (source_latents - latent_mean) / latent_std
        else:
            source_latents = precomputed_source_latents.float().contiguous()
            expected_shape = (
                batch_size,
                self.transformer.config.in_channels,
                height // self.latent_scale_factor,
                width // self.latent_scale_factor,
            )
            if tuple(source_latents.shape) == (1, *expected_shape[1:]) and batch_size > 1:
                source_latents = source_latents.repeat(batch_size, 1, 1, 1)
            if tuple(source_latents.shape) != expected_shape:
                raise ValueError(
                    "Expected `precomputed_source_latents` to have shape "
                    f"{expected_shape}, got {tuple(source_latents.shape)}."
                )
            source_latents = source_latents.repeat_interleave(num_images_per_prompt, dim=0)
        return source_latents, semantic_features

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: PipelineImageInput | None = None,
        generation_mode: str = "text",
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        precomputed_source_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 2048,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[["LLaDAImagePipeline", int, torch.Tensor, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> LLaDAImagePipelineOutput | tuple:
        r"""
        Generate images using text-only, VQ-conditioned, or editing inference.

        The timestep schedule is selected by the scheduler configuration. `use_uniform_sigmas=True` uses a uniform
        pre-shift grid; otherwise the source Kumaraswamy schedule is used.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                Text prompts that describe the generated image or requested edit.
            image (`PipelineImageInput`, *optional*):
                Reference image or image batch. Required in `"editing"` mode and rejected in other modes.
            generation_mode (`str`, defaults to `"text"`):
                Inference path. `"text"` uses only the text prompt. `"vq"` uses the MLLM to generate VQ tokens from
                the prompt at a maximum frontend resolution of 512 before diffusion. `"editing"` uses both
                reference-image SigVQ features and source-image latents.
            negative_prompt (`str` or `list[str]`, *optional*):
                Text excluded from generation. The checkpoint's empty CFG prompt is used by default.
            height (`int`, defaults to `1024`):
                Output image height.
            width (`int`, defaults to `1024`):
                Output image width.
            num_inference_steps (`int`, defaults to `20`):
                Number of flow-matching denoising steps.
            guidance_scale (`float`, defaults to `4.5`):
                Classifier-free guidance scale. Guidance is disabled at values up to `1.0`.
            num_images_per_prompt (`int`, defaults to `1`):
                Number of images generated per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Random generator or generator batch used to create the initial latents.
            latents (`torch.Tensor`, *optional*):
                Pre-generated patchified Flux2 latents.
            precomputed_source_latents (`torch.Tensor`, *optional*):
                Normalized, patchified reference-image latents produced by the VAE encoder. When provided in editing
                mode, the CPU VAE encode is skipped.
            prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed, projected positive prompt embeddings.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Valid-token mask for `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Precomputed, projected negative prompt embeddings.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Valid-token mask for `negative_prompt_embeds`.
            max_sequence_length (`int`, defaults to `2048`):
                Maximum text sequence length before the QueryFormer tokens are appended.
            output_type (`str`, defaults to `"pil"`):
                Output format. Choose `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict (`bool`, defaults to `True`):
                Whether to return [`LLaDAImagePipelineOutput`] instead of a tuple.
            callback_on_step_end (`Callable`, *optional*):
                Function called after each denoising step.
            callback_on_step_end_tensor_inputs (`list[str]`, defaults to `["latents"]`):
                Tensor names forwarded to `callback_on_step_end`.

        Returns:
            [`LLaDAImagePipelineOutput`] or `tuple`:
                Generated images or final patchified latents.
        """
        self.check_inputs(
            prompt,
            image,
            generation_mode,
            height,
            width,
            num_images_per_prompt,
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs,
            num_inference_steps,
        )

        if prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        elif isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)
        if self.dit_session is not None:
            # The reference DiT is meta/offloaded after export. AIC sessions
            # consume host tensors and perform the device transfer internally.
            device = torch.device("cpu")
            runtime_dtype = torch.float32
        else:
            device = self.transformer.device
            runtime_dtype = self.transformer.dtype
        self._guidance_scale = guidance_scale
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                num_images_per_prompt,
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                max_sequence_length,
                device,
            )
        )
        effective_batch_size = batch_size * num_images_per_prompt

        source_latents = None
        semantic_features = None
        if generation_mode == "vq":
            vq_token_ids = self.generate_vq_tokens(prompt, height, width)
            vq_token_ids = vq_token_ids.repeat_interleave(num_images_per_prompt, dim=0)
            if self.sigvq_tokens_session is None:
                with _component_timing(self.component_timings, "component.sigvq_tokens"):
                    semantic_features = self.sigvq(token_ids=vq_token_ids.to(self.sigvq.device)).semantic_features
            else:
                with _component_timing(self.component_timings, "component.sigvq_tokens"):
                    semantic_features = self.sigvq_tokens_session.run(vq_token_ids)
        elif generation_mode == "editing":
            source_latents, semantic_features = self._encode_source_image(
                image,
                height,
                width,
                batch_size,
                num_images_per_prompt,
                precomputed_source_latents,
            )

        latent_shape = (
            effective_batch_size,
            self.transformer.config.in_channels,
            height // self.latent_scale_factor,
            width // self.latent_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.float32)
            latents = latents.to(runtime_dtype).float()
        else:
            if latents.shape != latent_shape:
                raise ValueError(f"Expected `latents` to have shape {latent_shape}, got {tuple(latents.shape)}.")
            latents = latents.to(device=device, dtype=torch.float32)

        if self.scheduler.config.get("use_uniform_sigmas", False):
            # diffusers 0.39.0 does not natively support this scheduler option. Supplying the pre-shift grid
            # explicitly preserves the behavior of the patched scheduler used by LLaDA-Image-SGLang.
            sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)[:-1].tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        else:
            schedule_steps = num_inference_steps + 1
            schedule = torch.linspace(0.001, 1.0, schedule_steps, dtype=torch.float64)[:-1]
            schedule = (1 - (1 - schedule**1.17) ** 0.8) ** 1.1
            sigmas = (1 - schedule).tolist()
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        cond_cap_feats = [
            embeds[mask].to(device=device, dtype=runtime_dtype)
            for embeds, mask in zip(prompt_embeds, prompt_attention_mask.bool())
        ]
        if do_classifier_free_guidance:
            uncond_cap_feats = [
                embeds[mask].to(device=device, dtype=runtime_dtype)
                for embeds, mask in zip(negative_prompt_embeds, negative_prompt_attention_mask.bool())
            ]
            cap_feats = cond_cap_feats + uncond_cap_feats
        else:
            cap_feats = cond_cap_feats

        aic_cond_cap_feats = None
        aic_cond_cap_mask = None
        aic_uncond_cap_feats = None
        aic_uncond_cap_mask = None
        if self.dit_session is not None:
            if effective_batch_size != 1:
                raise ValueError("The initial LLaDA-Image DiT QPC supports batch size one")
            aic_cond_cap_feats, aic_cond_cap_mask = self._pad_condition_features(
                prompt_embeds,
                prompt_attention_mask,
                self.dit_session.caption_length,
            )
            if do_classifier_free_guidance:
                aic_uncond_cap_feats, aic_uncond_cap_mask = self._pad_condition_features(
                    negative_prompt_embeds,
                    negative_prompt_attention_mask,
                    self.dit_session.caption_length,
                )

        glm_cap_feats = None
        source_latent_list = None
        aic_semantic_features = None
        aic_semantic_mask = None
        aic_source_latents = None
        if semantic_features is not None:
            cond_glm_cap_feats = [
                features.to(device=device, dtype=runtime_dtype)
                for features in semantic_features
            ]
            if source_latents is not None:
                source_latent_list = [
                    latent.unsqueeze(1).to(device=device, dtype=runtime_dtype)
                    for latent in source_latents
                ]
            if do_classifier_free_guidance:
                empty_glm = semantic_features.new_zeros((0, semantic_features.shape[-1])).to(
                    device=device, dtype=runtime_dtype
                )
                glm_cap_feats = cond_glm_cap_feats + [empty_glm] * effective_batch_size
                if source_latent_list is not None:
                    source_latent_list = source_latent_list + source_latent_list
            else:
                glm_cap_feats = cond_glm_cap_feats
            if self.dit_session is not None:
                semantic_mask = torch.ones(
                    semantic_features.shape[:2], dtype=torch.int64, device=semantic_features.device
                )
                aic_semantic_features, aic_semantic_mask = self._pad_condition_features(
                    semantic_features,
                    semantic_mask,
                    self.dit_session.semantic_length,
                )
                if source_latents is not None:
                    aic_source_latents = source_latents.float().contiguous()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                with _component_timing(self.component_timings, "component.denoising.host_step_preparation"):
                    latent_model_input = (
                        torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
                    )
                    latent_list = [latent.unsqueeze(1).to(runtime_dtype) for latent in latent_model_input]
                    model_timestep = (timestep / self.scheduler.config.num_train_timesteps).expand(
                        latent_model_input.shape[0]
                    )

                if self.dit_session is not None:
                    if do_classifier_free_guidance:
                        single_timestep = (timestep / self.scheduler.config.num_train_timesteps).expand(
                            latents.shape[0]
                        )
                        with _component_timing(self.component_timings, "component.dit"):
                            conditional_output = self.dit_session.run(
                                latents,
                                single_timestep.to(runtime_dtype),
                                aic_cond_cap_feats,
                                aic_cond_cap_mask,
                                aic_semantic_features,
                                aic_semantic_mask,
                                aic_source_latents,
                            )
                            unconditional_output = self.dit_session.run(
                                latents,
                                single_timestep.to(runtime_dtype),
                                aic_uncond_cap_feats,
                                aic_uncond_cap_mask,
                                aic_semantic_features,
                                aic_semantic_mask,
                                aic_source_latents,
                            )
                        model_output = torch.cat([conditional_output, unconditional_output], dim=0)
                    else:
                        with _component_timing(self.component_timings, "component.dit"):
                            model_output = self.dit_session.run(
                                latent_model_input,
                                model_timestep.to(runtime_dtype),
                                aic_cond_cap_feats,
                                aic_cond_cap_mask,
                                aic_semantic_features,
                                aic_semantic_mask,
                                aic_source_latents,
                            )
                else:
                    with _component_timing(self.component_timings, "component.dit"):
                        model_output = self.transformer(
                            x=latent_list,
                            t=model_timestep.to(runtime_dtype),
                            cap_feats=cap_feats,
                            glm_cap_feats=glm_cap_feats,
                            source_latents=source_latent_list,
                        ).sample
                    model_output = torch.stack(model_output, dim=0).squeeze(2).float()
                with _component_timing(self.component_timings, "component.denoising.host_output_postprocessing"):
                    model_output = -model_output.float()

                    if do_classifier_free_guidance:
                        conditional_output, unconditional_output = model_output.chunk(2)
                        model_output = unconditional_output + self.guidance_scale * (
                            conditional_output - unconditional_output
                        )

                with _component_timing(self.component_timings, "component.scheduler"):
                    latents = self.scheduler.step(model_output, timestep, latents, return_dict=False)[0]

                with _component_timing(self.component_timings, "component.denoising.host_callback_and_progress"):
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for name in callback_on_step_end_tensor_inputs:
                            callback_kwargs[name] = locals()[name]
                        callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

        if output_type == "latent":
            images = latents
        else:
            if self.vae_decoder_session is not None:
                vae_device = torch.device("cpu")
                vae_dtype = torch.float32
            else:
                vae_device = self.vae.device
                vae_dtype = self.vae.dtype
            latents = latents.to(device=vae_device, dtype=vae_dtype)
            latent_mean = self._vae_latent_mean.view(1, -1, 1, 1).to(latents)
            latent_std = torch.sqrt(self._vae_latent_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents)
            latents = latents * latent_std + latent_mean
            latents = self._unpatchify_latents(latents)
            with _component_timing(self.component_timings, "component.vae_decoder"):
                if self.vae_decoder_session is None:
                    images = self.vae.decode(latents, return_dict=False)[0]
                else:
                    images = self.vae_decoder_session.run(latents)
            images = self.image_processor.postprocess(images, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return LLaDAImagePipelineOutput(images=images)


class QEffLLaDAImagePipeline:
    """QEfficient deployment wrapper around the reference LLaDA-Image pipeline.

    The reference :class:`LLaDAImagePipeline` remains responsible for model
    semantics and CPU fallback execution. This wrapper owns exportable modules,
    compilation configuration, and optional AI100 runtime sessions, following
    the composition pattern used by the Flux and Wan QEfficient pipelines.
    """

    _reference_pipeline_cls = LLaDAImagePipeline

    @staticmethod
    def get_default_config_path() -> str:
        """Return the packaged default LLaDA-Image compilation configuration."""
        return str(Path(__file__).resolve().parents[1] / "configs" / "llada_image_config.json")
    _SESSION_NAMES = {
        "queryformer_session",
        "composite_text_encoder_session",
        "text_projection_session",
        "sigvq_tokens_session",
        "llada2_text_session",
        "llada2_vq_session",
        "dit_session",
        "vae_encoder_session",
        "vae_decoder_session",
    }
    _DELEGATED_RUNTIME_NAMES = _SESSION_NAMES | {"component_timings", "stage_device_sessions"}

    def __init__(
        self,
        model: LLaDAImagePipeline | None = None,
        *,
        scheduler=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        queryformer=None,
        text_projection=None,
        sigvq=None,
        transformer=None,
        queryformer_session=None,
        composite_text_encoder_session=None,
        text_projection_session=None,
        sigvq_tokens_session=None,
        llada2_text_session=None,
        llada2_vq_session=None,
        dit_session=None,
        vae_encoder_session=None,
        vae_decoder_session=None,
        modules: Mapping[str, object] | None = None,
        composite_text_encoder=None,
    ):
        if model is None:
            required_components = {
                "scheduler": scheduler,
                "vae": vae,
                "text_encoder": text_encoder,
                "tokenizer": tokenizer,
                "queryformer": queryformer,
                "text_projection": text_projection,
                "sigvq": sigvq,
                "transformer": transformer,
            }
            missing = [name for name, component in required_components.items() if component is None]
            if missing:
                raise ValueError(
                    "Provide either a reference LLaDAImagePipeline or all pipeline components; "
                    f"missing: {', '.join(missing)}"
                )
            model = self._reference_pipeline_cls(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                queryformer=queryformer,
                text_projection=text_projection,
                sigvq=sigvq,
                transformer=transformer,
                queryformer_session=queryformer_session,
                composite_text_encoder_session=composite_text_encoder_session,
                text_projection_session=text_projection_session,
                sigvq_tokens_session=sigvq_tokens_session,
                llada2_text_session=llada2_text_session,
                llada2_vq_session=llada2_vq_session,
                dit_session=dit_session,
                vae_encoder_session=vae_encoder_session,
                vae_decoder_session=vae_decoder_session,
            )

        if not isinstance(model, self._reference_pipeline_cls):
            raise TypeError(
                "model must be an instance of LLaDAImagePipeline, "
                f"got {type(model).__name__}"
            )

        self.model = model
        self.modules = dict(modules or {})
        self.composite_text_encoder = composite_text_encoder
        self.sessions = {}
        self.custom_config = None
        self.onnx_paths = {}
        self.set_aic_sessions(
            queryformer_session=queryformer_session,
            composite_text_encoder_session=composite_text_encoder_session,
            text_projection_session=text_projection_session,
            sigvq_tokens_session=sigvq_tokens_session,
            llada2_text_session=llada2_text_session,
            llada2_vq_session=llada2_vq_session,
            dit_session=dit_session,
            vae_encoder_session=vae_encoder_session,
            vae_decoder_session=vae_decoder_session,
        )

    def __setattr__(self, name, value):
        """Keep legacy runtime-session assignments compatible with the wrapper."""
        if name in self._DELEGATED_RUNTIME_NAMES and "model" in self.__dict__:
            setattr(self.__dict__["model"], name, value)
            if name in self._SESSION_NAMES and "sessions" in self.__dict__:
                if value is None:
                    self.sessions.pop(name, None)
                else:
                    self.sessions[name] = value
            return
        object.__setattr__(self, name, value)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        torch_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        composite_text_encoder=None,
        composite_text_encoder_implementation: str | None = None,
    ) -> "QEffLLaDAImagePipeline":
        """Load the reference pipeline and wrap it for QEfficient deployment."""
        model = cls._reference_pipeline_cls.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        if composite_text_encoder is not None and composite_text_encoder_implementation is not None:
            raise ValueError(
                "Pass either composite_text_encoder or composite_text_encoder_implementation, not both."
            )
        if composite_text_encoder_implementation is not None:
            from QEfficient.diffusers.pipelines.llada_image.composite_encoder_module import (
                QEffLLaDAImageCompositeTextEncoder,
            )

            model_path = getattr(model, "_llada_image_model_path", None)
            if model_path is None:
                raise RuntimeError("The resolved LLaDA-Image model directory is unavailable for fused encoder loading.")
            composite_text_encoder = QEffLLaDAImageCompositeTextEncoder.from_pretrained(
                model_path,
                llada2_implementation=composite_text_encoder_implementation,
                torch_dtype=torch_dtype or torch.float32,
                queryformer=model.queryformer,
                text_encoder=model.text_encoder,
                text_projection=model.text_projection,
            )
        return cls(model, composite_text_encoder=composite_text_encoder)

    def __getattr__(self, name):
        """Preserve convenient access to reference pipeline attributes."""
        model = self.__dict__.get("model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    @staticmethod
    def _qpc_exists(path: str | Path | None) -> bool:
        return path is not None and (Path(path).expanduser() / "programqpc.bin").is_file()

    @staticmethod
    def _load_json(path: str | Path) -> dict:
        with Path(path).expanduser().resolve().open(encoding="utf-8") as config_file:
            value = json.load(config_file)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}")
        return value

    @classmethod
    def _complete_qpc_paths(cls, qpc_paths: Mapping[str, str | Path | None]) -> bool:
        required = ("composite_text_encoder", "dit", "vae_decoder")
        return all(cls._qpc_exists(qpc_paths.get(name)) for name in required)

    @staticmethod
    def _runtime_device_ids(config: Mapping[str, object]) -> dict[str, object]:
        runtime = config.get("runtime", {})
        if isinstance(runtime, Mapping) and isinstance(runtime.get("device_ids"), Mapping):
            return dict(runtime["device_ids"])

        device_ids = {}
        modules = config.get("modules", {})
        if isinstance(modules, Mapping):
            for module_name in ("composite_text_encoder", "dit", "vae_decoder"):
                module_config = modules.get(module_name, {})
                if isinstance(module_config, Mapping):
                    execute = module_config.get("execute", {})
                    if isinstance(execute, Mapping) and execute.get("device_ids") is not None:
                        device_ids[module_name] = execute["device_ids"]
        return device_ids

    def _prepare_compiled_qpcs(
        self,
        *,
        compile_config: str | Path,
        work_root: str | Path,
        mode: str,
        height: int,
        width: int,
        sequence_length: int,
        caption_length: int,
        semantic_length: int,
        parallel_compile: bool,
        use_onnx_subfunctions: bool,
    ) -> tuple[dict[str, str], dict[str, object]]:
        if use_onnx_subfunctions:
            raise ValueError("The fused LLaDA-Image flow requires use_onnx_subfunctions=False")
        if mode != "text":
            raise ValueError("The Flux-style fused LLaDA-Image flow currently supports generation_mode='text' only")
        if self.composite_text_encoder is None:
            raise ValueError(
                "The fused flow requires composite_text_encoder_implementation='default' "
                "when loading QEffLLaDAImagePipeline."
            )

        config_path = Path(compile_config).expanduser().resolve()
        config = self._load_json(config_path)
        root = Path(work_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"

        manifest = None
        if manifest_path.is_file():
            try:
                candidate = self._load_json(manifest_path)
                if self._complete_qpc_paths(candidate.get("qpc_paths", {})):
                    manifest = candidate
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                manifest = None

        configured_qpcs = {}
        modules = config.get("modules", {})
        if isinstance(modules, Mapping):
            for module_name in ("composite_text_encoder", "dit", "vae_decoder"):
                module_config = modules.get(module_name, {})
                execute = module_config.get("execute", {}) if isinstance(module_config, Mapping) else {}
                if isinstance(execute, Mapping) and execute.get("qpc_path") is not None:
                    configured_qpcs[module_name] = execute["qpc_path"]
        if manifest is None and self._complete_qpc_paths(configured_qpcs):
            manifest = {
                "model_path": str(getattr(self.model, "_llada_image_model_path", "")),
                "config": str(config_path),
                "compiled": True,
                "use_onnx_subfunctions": False,
                "qpc_paths": {name: str(path) for name, path in configured_qpcs.items()},
                "runtime": config.get("runtime", {}),
            }

        self._ensure_modules(
            mode=mode,
            height=height,
            width=width,
            sequence_length=sequence_length,
            caption_length=caption_length,
            semantic_length=semantic_length,
        )

        if manifest is None:
            logger.info("No complete LLaDA-Image QPC manifest found; exporting and compiling modules")
            self.compile(
                compile_config=str(config_path),
                export_dir=root / "export",
                mode=mode,
                height=height,
                width=width,
                sequence_length=sequence_length,
                caption_length=caption_length,
                semantic_length=semantic_length,
                parallel=parallel_compile,
                use_onnx_subfunctions=False,
            )
            qpc_paths = {
                name: str(module.qpc_path)
                for name, module in self.modules.items()
                if name in {"composite_text_encoder", "dit", "vae_decoder"} and module.qpc_path is not None
            }
            if not self._complete_qpc_paths(qpc_paths):
                raise RuntimeError(f"Compilation did not produce all required QPCs: {qpc_paths}")
            manifest = {
                "model_path": str(getattr(self.model, "_llada_image_model_path", "")),
                "config": str(config_path),
                "compiled": True,
                "use_onnx_subfunctions": False,
                "qpc_paths": qpc_paths,
                "runtime": config.get("runtime", {}),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            logger.info("LLaDA-Image QPC manifest: %s", manifest_path)

        runtime = manifest.get("runtime", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        runtime = dict(runtime)
        runtime.setdefault("device_ids", self._runtime_device_ids(config))
        return (
            {name: str(path) for name, path in manifest["qpc_paths"].items()},
            runtime,
        )

    def __call__(
        self,
        *args,
        custom_config_path: str | Path | None = None,
        compile_config: str | Path | None = None,
        parallel_compile: bool = False,
        use_onnx_subfunctions: bool = False,
        work_root: str | Path | None = None,
        stage_device_sessions: bool | None = None,
        **kwargs,
    ):
        """Run LLaDA-Image with optional Flux-style export/compile configuration."""
        if custom_config_path is not None and compile_config is not None:
            raise ValueError("Pass only one of custom_config_path or compile_config")
        compile_config = custom_config_path if custom_config_path is not None else compile_config
        if compile_config is None:
            compile_config = self.get_default_config_path()
        if len(args) > 1:
            raise TypeError("The LLaDA-Image pipeline accepts at most one positional argument: prompt")
        if args:
            if "prompt" in kwargs:
                raise TypeError("Prompt was provided both positionally and by keyword")
            kwargs["prompt"] = args[0]

        generation_mode = kwargs.get("generation_mode", "text")
        height = int(kwargs.get("height", 1024))
        width = int(kwargs.get("width", 1024))
        max_sequence_length = int(kwargs.get("max_sequence_length", 256))
        caption_length = max_sequence_length + int(getattr(self.model.queryformer.config, "num_queries", 256))
        if max_sequence_length != 256 or caption_length != 512:
            raise ValueError("The fused LLaDA-Image encoder requires max_sequence_length=256 and caption_length=512")

        root = Path(work_root or os.environ.get("QEFF_HOME", ".qeff-llada-image")).expanduser().resolve()
        os.environ.setdefault("QEFF_HOME", str(root))
        qpc_paths, runtime = self._prepare_compiled_qpcs(
            compile_config=compile_config,
            work_root=root,
            mode=generation_mode,
            height=height,
            width=width,
            sequence_length=max_sequence_length,
            caption_length=caption_length,
            semantic_length=0,
            parallel_compile=parallel_compile,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )

        stage = runtime.get("stage_device_sessions", True) if stage_device_sessions is None else stage_device_sessions
        guidance_scale = float(kwargs.get("guidance_scale", 4.5))
        if stage and guidance_scale > 1.0:
            raise ValueError("stage_device_sessions currently supports guidance_scale <= 1.0 only")

        device_ids = runtime.get("device_ids", {})
        if not isinstance(device_ids, Mapping):
            device_ids = {}
        timings: dict[str, float] = {}
        self.model.component_timings = timings
        self.model.stage_device_sessions = bool(stage)
        self.close_sessions()

        try:
            if stage:
                self.load_qpcs(
                    {"composite_text_encoder": qpc_paths["composite_text_encoder"]},
                    mode=generation_mode,
                    height=height,
                    width=width,
                    sequence_length=max_sequence_length,
                    caption_length=caption_length,
                    device_ids=device_ids,
                )
                with _component_timing(timings, "AIC text conditioning before device swap"):
                    precomputed_prompt = self.model.encode_prompt(
                        kwargs.get("prompt"),
                        kwargs.get("negative_prompt"),
                        do_classifier_free_guidance=guidance_scale > 1.0,
                        num_images_per_prompt=int(kwargs.get("num_images_per_prompt", 1)),
                        prompt_embeds=kwargs.get("prompt_embeds"),
                        prompt_attention_mask=kwargs.get("prompt_attention_mask"),
                        negative_prompt_embeds=kwargs.get("negative_prompt_embeds"),
                        negative_prompt_attention_mask=kwargs.get("negative_prompt_attention_mask"),
                        max_sequence_length=max_sequence_length,
                        device=self.model._execution_device,
                    )
                self.close_sessions()
                with _component_timing(timings, "AIC DiT/VAE session initialization"):
                    self.load_qpcs(
                        {"dit": qpc_paths["dit"], "vae_decoder": qpc_paths["vae_decoder"]},
                        mode=generation_mode,
                        height=height,
                        width=width,
                        sequence_length=max_sequence_length,
                        caption_length=caption_length,
                        device_ids=device_ids,
                    )
                call_kwargs = dict(kwargs)
                call_kwargs.update(
                    {
                        "prompt": None,
                        "negative_prompt": None,
                        "prompt_embeds": precomputed_prompt[0],
                        "prompt_attention_mask": precomputed_prompt[1],
                        "negative_prompt_embeds": precomputed_prompt[2],
                        "negative_prompt_attention_mask": precomputed_prompt[3],
                    }
                )
            else:
                self.load_qpcs(
                    qpc_paths,
                    mode=generation_mode,
                    height=height,
                    width=width,
                    sequence_length=max_sequence_length,
                    caption_length=caption_length,
                    device_ids=device_ids,
                )
                call_kwargs = dict(kwargs)

            return self.model(**call_kwargs)
        finally:
            self.last_timings = timings
            self.close_sessions()

    def register_module(self, name: str, module: object) -> object:
        """Register an export/compile adapter under a pipeline module name."""
        if not name or not hasattr(module, "export") or not hasattr(module, "compile"):
            raise ValueError("A registered module must have non-empty name, export(), and compile() methods")
        self.modules[name] = module
        if name == "composite_text_encoder":
            self.composite_text_encoder = module
        return module

    def configure_modules(
        self,
        *,
        mode: str = "text",
        height: int = 1024,
        width: int = 1024,
        sequence_length: int | None = None,
        caption_length: int | None = None,
        semantic_length: int = 0,
        include_vae_encoder: bool = False,
        composite_text_encoder=None,
    ) -> dict[str, object]:
        """Create QEfficient adapters for one fixed generation contract.

        LLaDA-Image's DiT and SigVQ graphs depend on generation mode and
        resolution, so they are configured explicitly instead of sharing one
        mutable adapter across all specializations.
        """
        if mode not in {"text", "vq", "editing"}:
            raise ValueError(f"Unsupported LLaDA-Image mode: {mode}")
        if mode in {"vq", "editing"} and semantic_length <= 0:
            raise ValueError(f"{mode} mode requires a positive semantic_length")

        import copy

        from QEfficient.diffusers.pipelines.llada_image.component_module import QEffLLaDAImageComponent
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import (
            QEffLLaDAImageVAEDecoder,
            QEffLLaDAImageVAEEncoder,
        )
        from QEfficient.diffusers.pipelines.pipeline_module import QEffLLaDAImageTransformer

        composite_text_encoder = composite_text_encoder or self.composite_text_encoder
        if sequence_length is None:
            sequence_length = 256 if composite_text_encoder is not None else 512
        if composite_text_encoder is not None:
            from QEfficient.diffusers.pipelines.llada_image.composite_encoder_module import (
                CAPTION_LENGTH,
                PROMPT_LENGTH,
            )

            if int(sequence_length) != PROMPT_LENGTH:
                raise ValueError(
                    f"The fused LLaDA-Image encoder requires sequence_length={PROMPT_LENGTH}, "
                    f"got {sequence_length}."
                )
            caption_length = int(caption_length or CAPTION_LENGTH)
            if caption_length != CAPTION_LENGTH:
                raise ValueError(
                    f"The fused LLaDA-Image encoder requires caption_length={CAPTION_LENGTH}, "
                    f"got {caption_length}."
                )
            text_modules = {"composite_text_encoder": composite_text_encoder}
        else:
            caption_length = int(caption_length or sequence_length)
            text_modules = {
                "queryformer": QEffLLaDAImageComponent(
                    self.model.queryformer,
                    component="queryformer",
                ),
                "text_projection": QEffLLaDAImageComponent(
                    self.model.text_projection,
                    component="text_projector",
                ),
            }

        modules = {
            **text_modules,
            "dit": QEffLLaDAImageTransformer(
                self.model.transformer,
                mode=mode,
                latent_height=height // self.model.latent_scale_factor,
                latent_width=width // self.model.latent_scale_factor,
                caption_length=caption_length,
                semantic_length=semantic_length,
                dynamic_resolution=composite_text_encoder is not None and mode == "text",
            ),
            "vae_decoder": QEffLLaDAImageVAEDecoder(self.model.vae),
        }

        if mode == "vq":
            modules["sigvq_tokens"] = QEffLLaDAImageComponent(
                self.model.sigvq,
                component="sigvq_tokens",
                sequence_length=semantic_length,
            )
        elif mode == "editing":
            modules["sigvq_image"] = QEffLLaDAImageComponent(
                self.model.sigvq,
                component="sigvq_image",
                image_height=height // 2,
                image_width=width // 2,
            )

        if include_vae_encoder:
            # Encoder and decoder adapters replace the wrapped VAE's forward
            # contract, so they must not share one mutable VAE instance.
            modules["vae_encoder"] = QEffLLaDAImageVAEEncoder(copy.deepcopy(self.model.vae))

        self.modules = modules
        self.module_contract = {
            "mode": mode,
            "height": int(height),
            "width": int(width),
            "sequence_length": int(sequence_length),
            "caption_length": caption_length,
            "semantic_length": int(semantic_length),
        }
        return self.modules

    def _ensure_modules(self, **configure_kwargs) -> None:
        if not self.modules:
            self.configure_modules(**configure_kwargs)

    def export(
        self,
        export_dir: str | Path | None = None,
        *,
        mode: str = "text",
        height: int = 1024,
        width: int = 1024,
        sequence_length: int | None = None,
        caption_length: int | None = None,
        semantic_length: int = 0,
        offload_pt_weights: bool = True,
        use_onnx_subfunctions: bool = False,
        composite_text_encoder=None,
    ) -> str:
        """Export all configured LLaDA-Image adapters and return the root directory."""
        self._ensure_modules(
            mode=mode,
            height=height,
            width=width,
            sequence_length=sequence_length,
            caption_length=caption_length,
            semantic_length=semantic_length,
            composite_text_encoder=composite_text_encoder,
        )
        export_root = Path(export_dir or Path.cwd() / "llada_image_export")
        export_root.mkdir(parents=True, exist_ok=True)

        for module_name, module in self.modules.items():
            module_dir = export_root / module_name
            module_dir.mkdir(parents=True, exist_ok=True)
            if module_name == "composite_text_encoder":
                module.export(
                    export_dir=module_dir,
                    offload_pt_weights=offload_pt_weights,
                    use_onnx_subfunctions=use_onnx_subfunctions,
                )
            elif module_name in {"queryformer", "text_projection", "sigvq_tokens", "sigvq_image"}:
                export_kwargs = {
                    "export_dir": module_dir,
                    "batch_size": 1,
                    "sequence_length": semantic_length if module_name == "sigvq_tokens" else sequence_length,
                    "image_height": height // 2 if module_name == "sigvq_image" else height,
                    "image_width": width // 2 if module_name == "sigvq_image" else width,
                    "offload_pt_weights": offload_pt_weights,
                }
                module.export(**export_kwargs)
            elif module_name == "dit":
                module.export(
                    export_dir=str(module_dir),
                    batch_size=1,
                    offload_pt_weights=offload_pt_weights,
                    use_onnx_subfunctions=use_onnx_subfunctions,
                )
            elif module_name == "vae_encoder":
                inputs, dynamic_axes, output_names = module.get_onnx_params(height=height, width=width)
                module.export(
                    inputs=inputs,
                    dynamic_axes=dynamic_axes,
                    output_names=output_names,
                    export_dir=str(module_dir),
                    export_kwargs={"offload_pt_weights": offload_pt_weights},
                )
            elif module_name == "vae_decoder":
                inputs, dynamic_axes, output_names = module.get_onnx_params(
                    latent_height=height // self.model.latent_scale_factor,
                    latent_width=width // self.model.latent_scale_factor,
                )
                module.export(
                    inputs=inputs,
                    dynamic_axes=dynamic_axes,
                    output_names=output_names,
                    export_dir=str(module_dir),
                    export_kwargs={"offload_pt_weights": offload_pt_weights},
                )
            else:
                raise ValueError(f"No export contract registered for LLaDA-Image module '{module_name}'")
            self.onnx_paths[module_name] = str(module.onnx_path)

        return str(export_root)

    def compile(
        self,
        compile_config: str,
        *,
        export_dir: str | Path | None = None,
        mode: str = "text",
        height: int = 1024,
        width: int = 1024,
        sequence_length: int | None = None,
        caption_length: int | None = None,
        semantic_length: int = 0,
        parallel: bool = False,
        use_onnx_subfunctions: bool = False,
        composite_text_encoder=None,
    ) -> dict[str, str | None]:
        """Export if needed and compile configured modules from a JSON config."""
        from QEfficient.diffusers.pipelines.pipeline_utils import (
            compile_modules_parallel,
            compile_modules_sequential,
            config_manager,
            set_execute_params,
        )

        self._ensure_modules(
            mode=mode,
            height=height,
            width=width,
            sequence_length=sequence_length,
            caption_length=caption_length,
            semantic_length=semantic_length,
            composite_text_encoder=composite_text_encoder,
        )
        config_manager(self, config_source=str(compile_config), use_onnx_subfunctions=use_onnx_subfunctions)
        set_execute_params(self)

        if any(module.onnx_path is None for module in self.modules.values()):
            self.export(
                export_dir=export_dir,
                mode=mode,
                height=height,
                width=width,
                sequence_length=sequence_length,
                caption_length=caption_length,
                semantic_length=semantic_length,
                use_onnx_subfunctions=use_onnx_subfunctions,
                composite_text_encoder=composite_text_encoder,
            )

        if parallel:
            compile_modules_parallel(self.modules, self.custom_config)
        else:
            compile_modules_sequential(self.modules, self.custom_config)
        return {name: getattr(module, "qpc_path", None) for name, module in self.modules.items()}

    def set_aic_sessions(self, **sessions) -> None:
        """Install pre-created AIC sessions on the delegated reference pipeline."""
        for name, session in sessions.items():
            if name not in self._SESSION_NAMES:
                raise ValueError(f"Unsupported LLaDA-Image session: {name}")
            if session is not None:
                setattr(self.model, name, session)
                self.sessions[name] = session

    @staticmethod
    def _session_device_ids(device_ids, name: str):
        if isinstance(device_ids, Mapping):
            return device_ids.get(name)
        return device_ids

    def load_qpcs(
        self,
        qpc_paths: Mapping[str, str | Path],
        *,
        mode: str = "text",
        height: int = 1024,
        width: int = 1024,
        sequence_length: int | None = None,
        caption_length: int | None = None,
        semantic_length: int = 0,
        device_ids=None,
    ) -> None:
        """Create supported component/DiT sessions from QPC paths.

        LLaDA2 text and VQ sessions remain injectable through ``set_aic_sessions``
        because their runtime contracts are model-variant-specific.
        """
        from QEfficient.diffusers.pipelines.llada_image.pipeline_module import (
            LLaDAImageCompositeTextEncoderAICSession,
            LLaDAImageDenoisingAICSession,
            LLaDAImageQueryFormerAICSession,
            LLaDAImageSigVQTokensAICSession,
            LLaDAImageTextProjectorAICSession,
            LLaDAImageVAEDecoderAICSession,
            LLaDAImageVAEEncoderAICSession,
        )

        if sequence_length is None:
            sequence_length = 256 if qpc_paths.get("composite_text_encoder") is not None else 512
        caption_length = int(caption_length or sequence_length)
        if mode not in {"text", "vq", "editing"}:
            raise ValueError(f"Unsupported LLaDA-Image mode: {mode}")
        if qpc_paths.get("sigvq_tokens") is not None and semantic_length <= 0:
            raise ValueError("sigvq_tokens QPC loading requires a positive semantic_length")
        sessions = {}
        hidden_size = int(self.model.text_encoder.config.hidden_size)
        num_queries = int(self.model.queryformer.config.num_queries)
        projection_dim = int(self.model.text_projection.config.projection_dim)
        feature_dim = int(self.model.sigvq.config.semantic_embed_dim)

        if qpc_paths.get("composite_text_encoder") is not None:
            sessions["composite_text_encoder_session"] = LLaDAImageCompositeTextEncoderAICSession(
                qpc_paths["composite_text_encoder"],
                input_length=sequence_length,
                hidden_size=hidden_size,
                output_length=caption_length,
                projection_dim=projection_dim,
                num_queries=num_queries,
                device_ids=self._session_device_ids(device_ids, "composite_text_encoder"),
            )
        elif qpc_paths.get("queryformer") is not None:
            sessions["queryformer_session"] = LLaDAImageQueryFormerAICSession(
                qpc_paths["queryformer"],
                input_length=sequence_length,
                hidden_size=hidden_size,
                num_queries=num_queries,
                device_ids=self._session_device_ids(device_ids, "queryformer"),
            )

        if qpc_paths.get("text_projection") is not None:
            sessions["text_projection_session"] = LLaDAImageTextProjectorAICSession(
                qpc_paths["text_projection"],
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                projection_dim=projection_dim,
                device_ids=self._session_device_ids(device_ids, "text_projection"),
            )

        if qpc_paths.get("sigvq_tokens") is not None:
            sessions["sigvq_tokens_session"] = LLaDAImageSigVQTokensAICSession(
                qpc_paths["sigvq_tokens"],
                sequence_length=semantic_length,
                feature_dim=feature_dim,
                device_ids=self._session_device_ids(device_ids, "sigvq_tokens"),
            )

        if qpc_paths.get("dit") is not None:
            latent_height = height // self.model.latent_scale_factor
            latent_width = width // self.model.latent_scale_factor
            sessions["dit_session"] = LLaDAImageDenoisingAICSession(
                qpc_paths["dit"],
                mode=mode,
                latent_shape=(1, self.model.transformer.config.in_channels, latent_height, latent_width),
                caption_length=caption_length,
                semantic_length=semantic_length,
                rope_config=self.model.transformer.config,
                device_ids=self._session_device_ids(device_ids, "dit"),
            )

        if qpc_paths.get("vae_encoder") is not None:
            sessions["vae_encoder_session"] = LLaDAImageVAEEncoderAICSession(
                qpc_paths["vae_encoder"],
                output_shape=(1, int(getattr(self.model.vae.config, "latent_channels", 32)), height // 8, width // 8),
                device_ids=self._session_device_ids(device_ids, "vae_encoder"),
            )

        if qpc_paths.get("vae_decoder") is not None:
            sessions["vae_decoder_session"] = LLaDAImageVAEDecoderAICSession(
                qpc_paths["vae_decoder"],
                output_shape=(1, 3, height, width),
                device_ids=self._session_device_ids(device_ids, "vae_decoder"),
            )

        self.set_aic_sessions(**sessions)

    def close_sessions(self) -> None:
        """Deactivate all sessions owned by this wrapper."""
        for name, session in list(self.sessions.items()):
            close = getattr(session, "close", None)
            if close is not None:
                close()
            setattr(self.model, name, None)
        self.sessions.clear()
