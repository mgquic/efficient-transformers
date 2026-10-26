"""QEfficient LLaDA2-MoE text encoder implementation for LLaDA-Image."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from QEfficient.base.pytorch_transforms import ExternalModuleMapperTransform, ModuleMappingTransform, PytorchTransform
from QEfficient.customop import CustomRMSNormAIC
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    stack_expert_linears,
    silu_glu_mlp,
)

# The final LLaDA-Image path uses Hugging Face's remote-code classes.  The
# external mapper below applies the QEff replacements by class name, so this
# module no longer depends on a second in-tree reference implementation.
_OriginalLLaDA2MoeAttention = None
_OriginalLLaDA2MoeRMSNorm = None
_OriginalLLaDA2MoeSparseMoeBlock = None


# -----------------------------------------------------------------------------
# RoPE compatibility and cached positional embeddings
# -----------------------------------------------------------------------------

def _register_llada2_rope_compatibility() -> None:
    """Support the legacy ``rope_type=default`` name used by LLaDA2."""
    from transformers import modeling_rope_utils

    if "default" in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        return

    def compute_default_rope_parameters(config, device=None, seq_len=None, **_kwargs):
        del seq_len
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim)
        )
        return inv_freq, 1.0

    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


class QEffLLaDA2MoeRotaryEmbedding(nn.Module):
    """Static RoPE cache for the fixed-shape LLaDA2 encoder export graph.

    This mirrors the QEff LLaDA/Llama cache strategy for LLaDA2's standard,
    unscaled RoPE.  ``position_ids`` remain an input and select rows from the
    cache, so the graph preserves the upstream position-ID contract without
    computing frequency products, cosine, or sine at inference time.
    """

    def __init__(self, config, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_type = None if rope_scaling is None else rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type not in (None, "default"):
            raise ValueError(f"The cached LLaDA2 RoPE experiment supports only default RoPE, got {rope_type!r}")

        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 0))
        if max_position_embeddings <= 0:
            raise ValueError("LLaDA2 cached RoPE requires config.max_position_embeddings to be positive")

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(positions, inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", embeddings.cos().to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", embeddings.sin().to(dtype=dtype), persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            raise ValueError("LLaDA2 cached RoPE requires position_ids")
        positions = position_ids.to(dtype=torch.long)
        return self.cos_cached[positions].to(dtype=hidden_states.dtype), self.sin_cached[positions].to(
            dtype=hidden_states.dtype
        )


class LLaDA2MoeModelAdapter:
    """Resolve the common model structure of original and remote LLaDA2 checkpoints."""

    @staticmethod
    def language_model(model: nn.Module) -> nn.Module:
        model_container = getattr(model, "backbone", model)
        model_container = getattr(model_container, "model", model_container)
        return getattr(model_container, "language_model", model_container)

    @staticmethod
    def lm_head(model: nn.Module) -> nn.Module:
        model_container = getattr(model, "backbone", model)
        model_container = getattr(model_container, "model", model_container)
        language_model = getattr(model_container, "language_model", model_container)
        for candidate in (model, model_container, language_model):
            lm_head = getattr(candidate, "lm_head", None)
            if lm_head is not None:
                return lm_head
        raise AttributeError("The LLaDA2 model does not expose an lm_head")

    @classmethod
    def install_cached_rope(cls, model: nn.Module) -> bool:
        """Install the static RoPE cache after the upstream model has loaded."""
        language_model = cls.language_model(model)
        existing_rotary = getattr(language_model, "rotary_emb", None)
        if existing_rotary is None or isinstance(existing_rotary, QEffLLaDA2MoeRotaryEmbedding):
            return False

        reference = next(language_model.parameters())
        language_model.rotary_emb = QEffLLaDA2MoeRotaryEmbedding(
            language_model.config,
            device=reference.device,
            dtype=reference.dtype,
        )
        return True


# -----------------------------------------------------------------------------
# QEff mapped modules
# -----------------------------------------------------------------------------

class QEffLLaDA2MoeRMSNorm(_OriginalLLaDA2MoeRMSNorm or nn.Module):
    """LLaDA2 RMSNorm routed directly through QEff's compiler-known custom op."""

    forward = CustomRMSNormAIC.forward


class QEffLLaDA2MoeAttention(_OriginalLLaDA2MoeAttention or nn.Module):
    """Reference attention with fixed-shape, no-cache eager attention."""

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
        if repeats == 1:
            return hidden_states
        batch, heads, sequence, head_dim = hidden_states.shape
        return (
            hidden_states[:, :, None, :, :]
            .expand(batch, heads, repeats, sequence, head_dim)
            .reshape(batch, heads * repeats, sequence, head_dim)
        )

    @staticmethod
    def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
        first, second = hidden_states.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    @classmethod
    def _apply_rotary_pos_emb(
        cls,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply partial RoPE to Q/K, preserving non-rotary head channels."""
        rotary_dim = cos.shape[-1]
        query_rotary, query_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
        key_rotary, key_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
        query_embed = query_rotary * cos + cls._rotate_half(query_rotary) * sin
        key_embed = key_rotary * cos + cls._rotate_half(key_rotary) * sin
        return torch.cat((query_embed, query_pass), dim=-1), torch.cat((key_embed, key_pass), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        **kwargs,
    ):
        del position_ids, kwargs
        if past_key_value is not None or use_cache:
            raise ValueError("The LLaDA2 AI100 graph does not support a KV cache")
        batch_size, query_length, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, query_length, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        if getattr(self, "query_layernorm", None) is not None:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)
        value_states = value_states.transpose(1, 2)
        if position_embeddings is None:
            raise ValueError("LLaDA2 attention requires precomputed position_embeddings")
        cos, sin = position_embeddings
        query_states, key_states = QEffLLaDA2MoeAttention._apply_rotary_pos_emb(
            query_states,
            key_states,
            cos.unsqueeze(1),
            sin.unsqueeze(1),
        )
        key_states = QEffLLaDA2MoeAttention._repeat_kv(key_states, self.num_key_value_groups)
        value_states = QEffLLaDA2MoeAttention._repeat_kv(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.to(dtype=scores.dtype)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
        output = torch.matmul(weights, value_states).transpose(1, 2).contiguous().reshape(batch_size, query_length, -1)
        output = self.dense(output)
        return output, (weights if output_attentions else None), None


class QEffLLaDA2MoeSparseMoeBlock(
    QEffMoEBlockMixin,
    _OriginalLLaDA2MoeSparseMoeBlock or nn.Module,
):
    """Original LLaDA2 MoE block with canonical QEff dispatch."""

    _moe_return_router_logits = False
    _moe_flavour = MoEFlavour.DECODE_BMM
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def __qeff_init__(self) -> None:
        QEffMoEBlockMixin.__qeff_init__(self)
        experts = getattr(self, "experts", None)
        expert_count = getattr(experts, "num_experts", None)
        if expert_count is None and experts is not None:
            expert_count = len(experts)
        self.num_experts = int(getattr(self.gate, "num_experts", expert_count))
        self.top_k = int(getattr(self.gate, "top_k", getattr(self, "num_experts_per_tok", self.num_experts)))
        self.act_fn = F.silu

    def transform_weights(self):
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        experts = self.experts
        if hasattr(experts, "gate_proj"):
            self.moe_weights = build_canonical_expert_weights(
                gate=experts.gate_proj,
                up=experts.up_proj,
                down=experts.down_proj,
                fused=False,
                transpose_gate_up=True,
                transpose_down=True,
                clone=True,
            )
            delete_module_attrs(experts, "gate_proj", "up_proj", "down_proj")
        else:
            self.moe_weights = build_canonical_expert_weights(
                gate=stack_expert_linears(experts, lambda expert: expert.gate_proj.weight),
                up=stack_expert_linears(experts, lambda expert: expert.up_proj.weight),
                down=stack_expert_linears(experts, lambda expert: expert.down_proj.weight),
                fused=False,
                transpose_gate_up=False,
                transpose_down=False,
                clone=True,
            )
            self.experts = None
        self.weights_transformed = True
        return self.moe_weights

    def route(self, hidden_states: torch.Tensor):
        topk_indices, topk_weights, router_logits = self.gate(hidden_states)
        return (topk_indices, topk_weights.to(dtype=hidden_states.dtype)), router_logits

    def apply_shared_experts(self, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        shared_experts = getattr(self, "shared_experts", None)
        return output if shared_experts is None else output + shared_experts(residual)

    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=F.silu))

    def forward(self, hidden_states: torch.Tensor):
        return QEffMoEBlockMixin.forward(self, hidden_states)


class LLaDA2MoeModuleTransform(ModuleMappingTransform):
    """Map classes from the in-tree original implementation exactly."""

    _module_mapping = {
        original: replacement
        for original, replacement in (
            (_OriginalLLaDA2MoeRMSNorm, QEffLLaDA2MoeRMSNorm),
            (_OriginalLLaDA2MoeAttention, QEffLLaDA2MoeAttention),
            (_OriginalLLaDA2MoeSparseMoeBlock, QEffLLaDA2MoeSparseMoeBlock),
        )
        if original is not None
    }


class LLaDA2MoeExternalModuleTransform(ExternalModuleMapperTransform):
    """Map the same behavior when Transformers loaded remote-code classes."""

    _match_class_replace_method = {}
    _match_string_replace_method = {
        "LLaDA2MoeRMSNorm": {"forward": QEffLLaDA2MoeRMSNorm.forward},
        "LLaDA2MoERMSNorm": {"forward": QEffLLaDA2MoeRMSNorm.forward},
        "LLaDA2MoeAttention": {"forward": QEffLLaDA2MoeAttention.forward},
        "LLaDA2MoeSdpaAttention": {"forward": QEffLLaDA2MoeAttention.forward},
        "LLaDA2MoeSparseMoeBlock": {
            "forward": QEffLLaDA2MoeSparseMoeBlock.forward,
            "route": QEffLLaDA2MoeSparseMoeBlock.route,
            "transform_weights": QEffLLaDA2MoeSparseMoeBlock.transform_weights,
            "apply_shared_experts": QEffLLaDA2MoeSparseMoeBlock.apply_shared_experts,
            "moe_profile": QEffLLaDA2MoeSparseMoeBlock.moe_profile,
            "get_supported_moe_flavours": QEffMoEBlockMixin.get_supported_moe_flavours,
            "execute_moe_flavour": QEffMoEBlockMixin.execute_moe_flavour,
            "moe_dispatch": QEffMoEBlockMixin.moe_dispatch,
            "routing_input": QEffMoEBlockMixin.routing_input,
            "__qeff_init__": QEffLLaDA2MoeSparseMoeBlock.__qeff_init__,
            "_moe_return_router_logits": False,
            "_moe_flavour": MoEFlavour.DECODE_BMM,
            "supported_moe_flavours": QEffLLaDA2MoeSparseMoeBlock.supported_moe_flavours,
        },
    }


# -----------------------------------------------------------------------------
# Transform wiring
# -----------------------------------------------------------------------------

class LLaDA2MoeTransform(PytorchTransform):
    """Apply QEff behavior to either the original or remote LLaDA2 model."""

    # Keep these attributes public for existing callers and structure tests.
    _match_class_replace_method = LLaDA2MoeExternalModuleTransform._match_class_replace_method
    _match_string_replace_method = LLaDA2MoeExternalModuleTransform._match_string_replace_method

    @classmethod
    def apply(cls, model: nn.Module) -> tuple[nn.Module, bool]:
        model, mapped = LLaDA2MoeModuleTransform.apply(model)
        model, external_mapped = LLaDA2MoeExternalModuleTransform.apply(model)
        rope_replaced = LLaDA2MoeModelAdapter.install_cached_rope(model)
        if mapped or external_mapped:
            from QEfficient.transformers.models.pytorch_transforms import OptimizedMoETransform

            model, moe_transformed = OptimizedMoETransform.apply(model, prefill_only=False)
            return model, mapped or external_mapped or moe_transformed or rope_replaced
        return model, rope_replaced


# -----------------------------------------------------------------------------
# Fixed-shape encoder export graph
# -----------------------------------------------------------------------------

class LLaDA2EncoderGraph(nn.Module):
    """Fixed-shape graph around the remote LLaDA2 backbone.

    Token embedding lookup stays outside this graph. ``logit_positions`` is only
    consumed by the VQ graph and prevents exporting a full ``[S, vocab]`` tensor.
    """

    def __init__(self, backbone: nn.Module, output_logits: bool):
        super().__init__()
        self.backbone = backbone
        self.output_logits = bool(output_logits)
        self.config = backbone.config

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logit_positions: torch.Tensor | None = None,
    ):
        language_model = LLaDA2MoeModelAdapter.language_model(self.backbone)
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            output_router_logits=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        if not self.output_logits:
            return hidden_states
        if logit_positions is None:
            raise ValueError("logit_positions is required for the LLaDA2 VQ graph")
        positions = logit_positions.to(dtype=torch.long)
        selected_hidden = torch.gather(
            hidden_states,
            1,
            positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1]),
        )
        return hidden_states, LLaDA2MoeModelAdapter.lm_head(self.backbone)(selected_hidden)
