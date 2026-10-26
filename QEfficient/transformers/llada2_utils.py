"""Host runtime helpers for the fixed-shape LLaDA2 AI100 graphs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def llada2_block_attention_mask(sequence_length: int, block_length: int, dtype: torch.dtype) -> torch.Tensor:
    """Return the additive block-lower-triangular mask used by LLaDA2 VQ."""
    if sequence_length < 1 or block_length < 1 or sequence_length % block_length:
        raise ValueError("sequence_length must be a positive multiple of block_length")
    num_blocks = sequence_length // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, dtype=torch.bool))
    allowed = block_mask.repeat_interleave(block_length, 0).repeat_interleave(block_length, 1)
    mask_value = torch.finfo(dtype).min
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), mask_value, dtype=dtype))[None, None]


def restrict_vq_logits(logits: torch.Tensor, image_token_offset: int, codebook_size: int) -> torch.Tensor:
    """Keep only the raw vocabulary interval consumed by SigVQ."""
    if logits.ndim < 1:
        raise ValueError(f"VQ logits must have a vocabulary dimension, got shape {tuple(logits.shape)}")
    start = int(image_token_offset)
    stop = start + int(codebook_size)
    if start < 0 or stop > logits.shape[-1] or start >= stop:
        raise ValueError(
            f"VQ image-token interval [{start}, {stop}) does not fit vocabulary size {logits.shape[-1]}"
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError("LLaDA2 VQ logits contain non-finite values")
    restricted = torch.full_like(logits, torch.finfo(logits.dtype).min)
    restricted[..., start:stop] = logits[..., start:stop]
    return restricted


class LLaDA2AICSession:
    """Runtime adapter for text and VQ QPCs.

    Tokenizer and embedding lookup stay on CPU. The host also owns VQ CFG,
    confidence, remasking, block scheduling, and token commits.
    """

    def __init__(
        self,
        qpc_path: str,
        *,
        sequence_length: int,
        hidden_size: int,
        vocab_size: int = 173568,
        vq: bool = False,
        device_ids: list[int] | None = None,
        input_dtype: np.dtype = np.float16,
    ) -> None:
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.vq = bool(vq)
        self.input_dtype = np.dtype(input_dtype)
        self.session = QAICInferenceSession(qpc_path, device_ids=device_ids)
        self._binding_dtypes = {
            name: self._binding_dtype(name, self.input_dtype)
            for name in ("inputs_embeds", "attention_mask", "position_ids", "hidden_states", "logits")
        }
        buffers = {
            "hidden_states": np.zeros(
                (1, self.sequence_length, self.hidden_size), dtype=self._binding_dtypes["hidden_states"]
            )
        }
        if self.vq:
            buffers["logits"] = np.zeros(
                (1, 32, self.vocab_size), dtype=self._binding_dtypes["logits"]
            )
        self.session.set_buffers(buffers)

    def _binding_dtype(self, name: str, default) -> np.dtype:
        index = self.session.binding_index_map.get(name)
        if index is None:
            return np.dtype(default)
        binding = self.session.bindings[index]
        return self.session.aic_to_np_dtype_mapping[binding.type]

    def close(self) -> None:
        if self.session is not None:
            self.session.deactivate()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()

    def _run(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logit_positions: torch.Tensor | None = None,
    ):
        if tuple(inputs_embeds.shape[:2]) != (1, self.sequence_length):
            raise ValueError(
                "LLaDA2 AIC expects inputs_embeds with shape "
                f"(1, {self.sequence_length}, hidden), got {tuple(inputs_embeds.shape)}"
            )
        inputs = {
            "inputs_embeds": inputs_embeds.detach().cpu().numpy().astype(self._binding_dtypes["inputs_embeds"], copy=False),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(self._binding_dtypes["attention_mask"], copy=False),
            "position_ids": position_ids.detach().cpu().numpy().astype(self._binding_dtypes["position_ids"], copy=False),
        }
        if self.vq:
            if logit_positions is None or tuple(logit_positions.shape) != (1, 32):
                raise ValueError("VQ LLaDA2 AIC calls require logit_positions with shape (1, 32)")
            inputs["logit_positions"] = logit_positions.detach().cpu().numpy().astype(
                self._binding_dtypes.get("logit_positions", np.int64), copy=False
            )
        outputs = self.session.run(inputs)
        hidden_states = torch.from_numpy(np.array(outputs["hidden_states"], copy=True)).float()
        if not torch.isfinite(hidden_states).all():
            raise RuntimeError("LLaDA2 AIC returned non-finite hidden states")
        if not self.vq:
            return hidden_states
        logits = torch.from_numpy(np.array(outputs["logits"], copy=True)).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("LLaDA2 AIC returned non-finite logits")
        return hidden_states, logits

    def text(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.vq:
            raise RuntimeError("A VQ LLaDA2 session cannot execute the text graph")
        return self._run(inputs_embeds, attention_mask, position_ids)

    def vq_forward(
        self,
        input_ids: torch.Tensor,
        embedding_lookup,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        logit_positions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.vq:
            raise RuntimeError("A text LLaDA2 session cannot execute the VQ graph")
        inputs_embeds = embedding_lookup(input_ids).to(dtype=torch.float32)
        return self._run(inputs_embeds, attention_mask, position_ids, logit_positions)[1]

    @torch.no_grad()
    def generate_vq(
        self,
        input_ids: torch.Tensor,
        uncond_ids: torch.Tensor,
        embedding_lookup,
        *,
        block_length: int = 32,
        steps: int = 8,
        generation_length: int | None = None,
        cfg_scale: float = 2.0,
        image_token_offset: int | None = None,
        codebook_size: int | None = None,
        mask_id: int = 156895,
        eoi_id: int = 156902,
        temperature: float = 0.0,
        threshold: float = 0.95,
        eos_early_stop: bool = True,
    ) -> torch.Tensor:
        """Run upstream block diffusion while forwards execute on AI100."""
        if input_ids.shape[0] != 1 or uncond_ids.shape[0] != 1:
            raise ValueError("The initial LLaDA2 AIC CFG implementation supports batch size one")
        prompt_length = input_ids.shape[1]
        generation_length = (
            self.sequence_length - prompt_length if generation_length is None else int(generation_length)
        )
        if generation_length <= 0 or generation_length % block_length:
            raise ValueError("The VQ QPC sequence length must leave whole 32-token generation blocks")
        generation_end = prompt_length + generation_length
        if generation_end > self.sequence_length:
            raise ValueError("The requested VQ generation range exceeds the compiled LLaDA2 sequence length")
        num_blocks = (generation_end + block_length - 1) // block_length
        prefill_blocks = prompt_length // block_length
        if prefill_blocks < 1:
            raise ValueError("The VQ prompt must occupy at least one complete block")
        if steps < 1 or block_length % steps:
            raise ValueError("steps must divide the LLaDA2 VQ block length")
        schedule = torch.full((steps,), block_length // steps, dtype=torch.long)
        full_mask = llada2_block_attention_mask(self.sequence_length, block_length, torch.float16)
        position_ids = torch.arange(self.sequence_length, dtype=torch.long).unsqueeze(0)
        tokens = torch.full((1, self.sequence_length), mask_id, dtype=torch.long)
        tokens[:, :prompt_length] = input_ids.cpu()
        uncond = torch.full((1, prompt_length), mask_id, dtype=torch.long)
        uncond[:, -uncond_ids.shape[1] :] = uncond_ids.cpu()
        pad_len = prompt_length - uncond_ids.shape[1]
        uncond_mask = full_mask.clone()
        if pad_len:
            uncond_mask[:, :, :, :pad_len] = torch.finfo(torch.float16).min
        uncond_positions = torch.cat(
            [torch.zeros(pad_len, dtype=torch.long), torch.arange(self.sequence_length - pad_len, dtype=torch.long)]
        ).unsqueeze(0)

        for block_index in range(prefill_blocks, num_blocks):
            window_end = (block_index + 1) * block_length
            active_positions = torch.arange(window_end - block_length, window_end, dtype=torch.long).unsqueeze(0)
            for step_index in range(steps):
                active = tokens[:, active_positions[0]] == mask_id
                if not bool(active.any()):
                    break
                conditional_logits = self.vq_forward(
                    tokens, embedding_lookup, full_mask, position_ids, active_positions
                )
                if cfg_scale != 1.0:
                    unconditional_tokens = tokens.clone()
                    unconditional_tokens[:, :prompt_length] = uncond
                    unconditional_logits = self.vq_forward(
                        unconditional_tokens,
                        embedding_lookup,
                        uncond_mask,
                        uncond_positions,
                        active_positions,
                    )
                    active_logits = unconditional_logits + cfg_scale * (conditional_logits - unconditional_logits)
                else:
                    active_logits = conditional_logits
                if image_token_offset is not None or codebook_size is not None:
                    if image_token_offset is None or codebook_size is None:
                        raise ValueError("image_token_offset and codebook_size must be provided together")
                    active_logits = restrict_vq_logits(active_logits, image_token_offset, codebook_size)
                if temperature == 0.0:
                    probabilities = F.softmax(active_logits, dim=-1)
                    predicted = active_logits.argmax(dim=-1)
                    confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                else:
                    probabilities = F.softmax(active_logits / temperature, dim=-1)
                    predicted = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).view(1, -1)
                    confidence = probabilities.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                count = int(schedule[step_index])
                scores = torch.where(active, confidence, torch.full_like(confidence, -torch.inf))
                selected = scores > threshold
                if int(selected.sum()) < count:
                    _, indices = torch.topk(scores[0], k=min(count, int(active.sum())))
                    selected.zero_()
                    selected[0, indices] = True
                tokens[:, active_positions[0][selected[0]]] = predicted[selected]
                if eos_early_stop and bool((tokens[0, prompt_length:window_end] == eoi_id).any()):
                    stop_position = int((tokens[0, prompt_length:window_end] == eoi_id).nonzero()[0]) + prompt_length
                    if bool((tokens[0, prompt_length:stop_position] != mask_id).all()):
                        return tokens[:, : stop_position + 1]
        return tokens[:, :generation_end]
