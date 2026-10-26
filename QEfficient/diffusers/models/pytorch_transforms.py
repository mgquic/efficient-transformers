# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    WanDecoder3d,
    WanEncoder3d,
    WanResample,
    WanResidualBlock,
)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor, WanTransformer3DModel
from torch import nn
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop.rms_norm import CustomRMSNormAIC
from QEfficient.diffusers.models.autoencoders.autoencoder_kl_wan import (
    QEffAutoencoderKLWan,
    QEffWanDecoder3d,
    QEffWanEncoder3d,
    QEffWanResample,
    QEffWanResidualBlock,
)
from QEfficient.diffusers.models.normalization import (
    QEffAdaLayerNormContinuous,
    QEffAdaLayerNormZero,
    QEffAdaLayerNormZeroSingle,
)
from QEfficient.diffusers.models.transformers.transformer_flux import (
    QEffFluxAttention,
    QEffFluxAttnProcessor,
    QEffFluxSingleTransformerBlock,
    QEffFluxTransformer2DModel,
    QEffFluxTransformerBlock,
)
from QEfficient.diffusers.models.transformers.transformer_wan import (
    QEffWanAttention,
    QEffWanAttnProcessor,
    QEffWanTransformer3DModel,
)
from QEfficient.diffusers.models.transformers.transformer_llada_image import (
    LLaDAImageAttention,
    LLaDAImageQueryAttention,
    LLaDAImageQueryFormerBlock,
    LLaDAImageQueryFormerModel,
    LLaDAImageSigVQAttention,
    LLaDAImageSigVQModel,
    LLaDAImageSigVQVisionBlock,
    LLaDAImageTextProjectionAttention,
    LLaDAImageTextProjectionBlock,
    LLaDAImageTextProjectionModel,
    LLaDAImageTransformer2DModel,
    LLaDAImageTransformerBlock,
    QEffLLaDAImageAttention,
    QEffLLaDAImageQueryAttention,
    QEffLLaDAImageQueryFormerBlock,
    QEffLLaDAImageQueryFormerModel,
    QEffLLaDAImageSigVQAttention,
    QEffLLaDAImageSigVQModel,
    QEffLLaDAImageSigVQVisionBlock,
    QEffLLaDAImageTextProjectionAttention,
    QEffLLaDAImageTextProjectionBlock,
    QEffLLaDAImageTextProjectionModel,
    QEffLLaDAImageTransformer2DModel,
    QEffLLaDAImageTransformerBlock,
)
from QEfficient.transformers.models.clip.modeling_clip import QEffCLIPTextTransformer


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        RMSNorm: CustomRMSNormAIC,
        nn.RMSNorm: CustomRMSNormAIC,  #  for torch.nn.RMSNorm
    }


class CLIPTextTransform(ModuleMappingTransform):
    _module_mapping = {
        CLIPTextTransformer: QEffCLIPTextTransformer,
    }


class AttentionTransform(ModuleMappingTransform):
    _module_mapping = {
        FluxSingleTransformerBlock: QEffFluxSingleTransformerBlock,
        FluxTransformerBlock: QEffFluxTransformerBlock,
        FluxTransformer2DModel: QEffFluxTransformer2DModel,
        FluxAttention: QEffFluxAttention,
        FluxAttnProcessor: QEffFluxAttnProcessor,
        WanAttnProcessor: QEffWanAttnProcessor,
        WanAttention: QEffWanAttention,
        WanTransformer3DModel: QEffWanTransformer3DModel,
        AutoencoderKLWan: QEffAutoencoderKLWan,
        WanDecoder3d: QEffWanDecoder3d,
        WanEncoder3d: QEffWanEncoder3d,
        WanResidualBlock: QEffWanResidualBlock,
        WanResample: QEffWanResample,
    }


class LLaDAImageAttentionTransform(ModuleMappingTransform):
    """Replace LLaDA-Image attention with the QAIC-safe processor."""

    _module_mapping = {
        LLaDAImageAttention: QEffLLaDAImageAttention,
    }


class LLaDAImageTransformerTransform(ModuleMappingTransform):
    """Apply the QEff LLaDA-Image DiT class hierarchy before export."""

    _module_mapping = {
        LLaDAImageTransformer2DModel: QEffLLaDAImageTransformer2DModel,
        LLaDAImageTransformerBlock: QEffLLaDAImageTransformerBlock,
        LLaDAImageAttention: QEffLLaDAImageAttention,
    }


class LLaDAImageComponentTransform(ModuleMappingTransform):
    """Apply the QEff class hierarchy to exportable LLaDA-Image components."""

    _module_mapping = {
        LLaDAImageQueryFormerModel: QEffLLaDAImageQueryFormerModel,
        LLaDAImageQueryFormerBlock: QEffLLaDAImageQueryFormerBlock,
        LLaDAImageQueryAttention: QEffLLaDAImageQueryAttention,
        LLaDAImageTextProjectionModel: QEffLLaDAImageTextProjectionModel,
        LLaDAImageTextProjectionBlock: QEffLLaDAImageTextProjectionBlock,
        LLaDAImageTextProjectionAttention: QEffLLaDAImageTextProjectionAttention,
        LLaDAImageSigVQModel: QEffLLaDAImageSigVQModel,
        LLaDAImageSigVQVisionBlock: QEffLLaDAImageSigVQVisionBlock,
        LLaDAImageSigVQAttention: QEffLLaDAImageSigVQAttention,
    }


class NormalizationTransform(ModuleMappingTransform):
    _module_mapping = {
        AdaLayerNormZero: QEffAdaLayerNormZero,
        AdaLayerNormZeroSingle: QEffAdaLayerNormZeroSingle,
        AdaLayerNormContinuous: QEffAdaLayerNormContinuous,
    }
