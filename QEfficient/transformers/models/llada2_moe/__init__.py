"""QEfficient support for the LLaDA2 MoE encoder."""

from .modeling_llada2_moe import (
    LLaDA2EncoderGraph,
    LLaDA2MoeExternalModuleTransform,
    LLaDA2MoeModuleTransform,
    LLaDA2MoeTransform,
    QEffLLaDA2MoeAttention,
    QEffLLaDA2MoeRMSNorm,
    QEffLLaDA2MoeSparseMoeBlock,
)

# Preserve the public helper names as aliases to the mapped module methods.
qeff_llada2_moe_block_forward = QEffLLaDA2MoeSparseMoeBlock.forward
qeff_llada2_moe_block_init = QEffLLaDA2MoeSparseMoeBlock.__qeff_init__
qeff_llada2_moe_block_route = QEffLLaDA2MoeSparseMoeBlock.route
qeff_llada2_moe_block_transform_weights = QEffLLaDA2MoeSparseMoeBlock.transform_weights
qeff_llada2_moe_rms_norm_forward = QEffLLaDA2MoeRMSNorm.forward

__all__ = [
    "LLaDA2EncoderGraph",
    "LLaDA2MoeExternalModuleTransform",
    "LLaDA2MoeModuleTransform",
    "LLaDA2MoeTransform",
    "QEffLLaDA2MoeAttention",
    "QEffLLaDA2MoeRMSNorm",
    "QEffLLaDA2MoeSparseMoeBlock",
    "qeff_llada2_moe_block_forward",
    "qeff_llada2_moe_block_init",
    "qeff_llada2_moe_block_route",
    "qeff_llada2_moe_block_transform_weights",
    "qeff_llada2_moe_rms_norm_forward",
]
