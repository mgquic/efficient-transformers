# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import time
from contextlib import contextmanager

import torch

from QEfficient.diffusers.models.transformers.transformer_llada_image import build_llada_image_text_rope
from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE
from QEfficient.utils import constants


logger = logging.getLogger(__name__)


@contextmanager
def _session_timing(timings: dict[str, float] | None, name: str):
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


class QEffLLaDAImageVAEEncoder(QEffVAE):
    """QEfficient VAE encoder adapter for the LLaDA-Image Flux2 VAE."""

    def __init__(self, model):
        super().__init__(model, "encoder")

        def encode_forward(image):
            return self.model.encode(image).latent_dist.mode()

        self.model.forward = encode_forward

    def get_onnx_params(self, height: int = 256, width: int = 256):
        """Return the 4D image-to-latent contract used by LLaDA-Image."""
        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        example_inputs = {"image": torch.randn(batch_size, 3, height, width)}
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "latents": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        }
        return example_inputs, dynamic_axes, ["latents"]


class QEffLLaDAImageVAEDecoder(QEffVAE):
    """QEfficient VAE decoder adapter for the LLaDA-Image Flux2 VAE."""

    def __init__(self, model):
        super().__init__(model, "decoder")

        def decode_forward(latent_sample, return_dict=False):
            return self.model.decode(latent_sample, return_dict=return_dict)

        self.model.forward = decode_forward


class _LLaDAImageVAESession:
    """Small runtime adapter shared by the optional VAE encoder/decoder sessions."""

    def __init__(self, qpc_path, *, input_name: str, output_name: str, output_shape: tuple[int, ...], device_ids=None):
        import numpy as np

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self._np = np
        self.input_name = input_name
        self.output_name = output_name
        self.output_shape = tuple(int(value) for value in output_shape)
        self.session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
        self.session.set_buffers({output_name: np.zeros(self.output_shape, dtype=self._binding_dtype(output_name))})

    def _binding_dtype(self, name: str):
        index = self.session.binding_index_map.get(name)
        if index is None:
            return self._np.dtype(self._np.float32)
        return self.session.aic_to_np_dtype_mapping[self.session.bindings[index].type]

    def run(self, values: torch.Tensor) -> torch.Tensor:
        values = self._np.asarray(values.detach().cpu(), dtype=self._binding_dtype(self.input_name))
        output = self.session.run({self.input_name: values})[self.output_name]
        output = self._np.array(output, copy=True)
        if tuple(output.shape) != self.output_shape:
            raise RuntimeError(
                f"{self.__class__.__name__} returned {tuple(output.shape)}, expected {self.output_shape}"
            )
        if not self._np.isfinite(output).all():
            raise RuntimeError(f"{self.__class__.__name__} returned non-finite output")
        return torch.from_numpy(output).float()

    def close(self):
        if self.session is not None:
            self.session.deactivate()
            self.session = None


class LLaDAImageVAEEncoderAICSession(_LLaDAImageVAESession):
    """Run a fixed-shape LLaDA-Image VAE encoder QPC on AI100."""

    def __init__(self, qpc_path, *, output_shape: tuple[int, ...], device_ids=None):
        super().__init__(
            qpc_path,
            input_name="image",
            output_name="latents",
            output_shape=output_shape,
            device_ids=device_ids,
        )


class LLaDAImageVAEDecoderAICSession(_LLaDAImageVAESession):
    """Run a fixed-shape LLaDA-Image VAE decoder QPC on AI100."""

    def __init__(self, qpc_path, *, output_shape: tuple[int, ...], device_ids=None):
        super().__init__(
            qpc_path,
            input_name="latent_sample",
            output_name="sample",
            output_shape=output_shape,
            device_ids=device_ids,
        )


class _LLaDAImageComponentAICSession:
    """Runtime adapter for a fixed-shape tensor-only LLaDA-Image QPC."""

    def __init__(self, qpc_path, *, output_name: str, output_shape: tuple[int, ...], device_ids=None):
        import numpy as np

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self._np = np
        self.session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
        self.output_name = output_name
        self.output_shape = tuple(int(value) for value in output_shape)
        self.output_dtype = self._binding_dtype(output_name, np.float32)
        self.session.set_buffers({output_name: np.zeros(self.output_shape, dtype=self.output_dtype)})

    def _binding_dtype(self, name: str, default):
        index = self.session.binding_index_map.get(name)
        if index is None:
            return self._np.dtype(default)
        binding = self.session.bindings[index]
        return self.session.aic_to_np_dtype_mapping[binding.type]

    def _run(self, inputs):
        output = self.session.run(inputs)[self.output_name]
        output = self._np.array(output, copy=True)
        if tuple(output.shape) != self.output_shape:
            raise RuntimeError(
                f"{self.__class__.__name__} returned {tuple(output.shape)}, expected {self.output_shape}"
            )
        if not self._np.isfinite(output).all():
            raise RuntimeError(f"{self.__class__.__name__} returned non-finite output")
        return output

    def close(self):
        if self.session is not None:
            self.session.deactivate()
            self.session = None


class LLaDAImageQueryFormerAICSession(_LLaDAImageComponentAICSession):
    """Run a fixed-shape QueryFormer graph on AI100."""

    def __init__(self, qpc_path, *, input_length: int, hidden_size: int, num_queries: int, device_ids=None):
        self.input_length = int(input_length)
        self.hidden_size = int(hidden_size)
        self.num_queries = int(num_queries)
        super().__init__(
            qpc_path,
            output_name="query_embeds",
            output_shape=(1, self.num_queries, self.hidden_size),
            device_ids=device_ids,
        )
        binding_index = self.session.binding_index_map.get("inputs_embeds")
        if binding_index is not None:
            binding_dims = tuple(int(dim) for dim in self.session.bindings[binding_index].dims)
            if len(binding_dims) == 3 and binding_dims[0] == 1 and binding_dims[2] == self.hidden_size:
                qpc_input_length = binding_dims[1]
                if qpc_input_length != self.input_length:
                    logger.info(
                        "QueryFormer runtime length %d differs from requested logical length %d; "
                        "using the QPC compiled length",
                        qpc_input_length,
                        self.input_length,
                    )
                self.input_length = qpc_input_length

    def run(self, inputs_embeds, attention_mask):
        inputs_embeds = self._np.asarray(
            inputs_embeds.detach().cpu(), dtype=self._binding_dtype("inputs_embeds", self._np.float32)
        )
        attention_mask = self._np.asarray(
            attention_mask.detach().cpu(), dtype=self._binding_dtype("attention_mask", self._np.int8)
        )
        expected_shape = (1, self.input_length, self.hidden_size)
        if tuple(inputs_embeds.shape) != expected_shape:
            raise ValueError(f"QueryFormer QPC expects {expected_shape}, got {tuple(inputs_embeds.shape)}")
        if tuple(attention_mask.shape) != (1, self.input_length):
            raise ValueError(
                f"QueryFormer QPC expects attention_mask shape {(1, self.input_length)}, got {tuple(attention_mask.shape)}"
            )
        return torch.from_numpy(self._run({"inputs_embeds": inputs_embeds, "attention_mask": attention_mask})).float()


class LLaDAImageCompositeTextEncoderAICSession(_LLaDAImageComponentAICSession):
    """Run the fused QueryFormer -> LLaDA2 -> text-projector graph on AI100."""

    def __init__(
        self,
        qpc_path,
        *,
        input_length: int,
        hidden_size: int,
        output_length: int,
        projection_dim: int,
        num_queries: int,
        device_ids=None,
    ):
        self.input_length = int(input_length)
        self.hidden_size = int(hidden_size)
        self.output_length = int(output_length)
        self.projection_dim = int(projection_dim)
        self.num_queries = int(num_queries)
        super().__init__(
            qpc_path,
            output_name="prompt_embeds",
            output_shape=(1, self.output_length, self.projection_dim),
            device_ids=device_ids,
        )

    def run(self, inputs_embeds, attention_mask):
        inputs_embeds = self._np.asarray(
            inputs_embeds.detach().cpu(), dtype=self._binding_dtype("inputs_embeds", self._np.float32)
        )
        attention_mask = self._np.asarray(
            attention_mask.detach().cpu(), dtype=self._binding_dtype("attention_mask", self._np.bool_)
        )
        expected_embeds = (1, self.input_length, self.hidden_size)
        expected_mask = (1, self.input_length)
        if tuple(inputs_embeds.shape) != expected_embeds:
            raise ValueError(f"Composite text encoder QPC expects {expected_embeds}, got {tuple(inputs_embeds.shape)}")
        if tuple(attention_mask.shape) != expected_mask:
            raise ValueError(f"Composite text encoder QPC expects {expected_mask}, got {tuple(attention_mask.shape)}")
        output = self._run({"inputs_embeds": inputs_embeds, "attention_mask": attention_mask})
        return torch.from_numpy(output).float()


class LLaDAImageTextProjectorAICSession(_LLaDAImageComponentAICSession):
    """Run the fixed 512-token text connector/projector graph on AI100."""

    def __init__(
        self,
        qpc_path,
        *,
        sequence_length: int,
        hidden_size: int,
        projection_dim: int,
        device_ids=None,
        defer_activation: bool = False,
    ):
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.projection_dim = int(projection_dim)
        self._deferred_qpc = None
        if defer_activation:
            self.output_name = "projected_hidden_states"
            self.output_shape = (1, self.sequence_length, self.projection_dim)
            self.output_dtype = None
            self.session = None
            self._deferred_qpc = (str(qpc_path), device_ids)
            self.has_attention_mask = None
            return
        super().__init__(
            qpc_path,
            output_name="projected_hidden_states",
            output_shape=(1, self.sequence_length, self.projection_dim),
            device_ids=device_ids,
        )
        self.has_attention_mask = "attention_mask" in self.session.binding_index_map

    def _activate_deferred(self):
        if self._deferred_qpc is None or self.session is not None:
            return
        qpc_path, device_ids = self._deferred_qpc
        super().__init__(
            qpc_path,
            output_name="projected_hidden_states",
            output_shape=(1, self.sequence_length, self.projection_dim),
            device_ids=device_ids,
        )
        self.has_attention_mask = "attention_mask" in self.session.binding_index_map
        self._deferred_qpc = None

    def run(self, hidden_states, attention_mask=None):
        self._activate_deferred()
        hidden_states = self._np.asarray(
            hidden_states.detach().cpu(), dtype=self._binding_dtype("hidden_states", self._np.float32)
        )
        expected_shape = (1, self.sequence_length, self.hidden_size)
        if tuple(hidden_states.shape) != expected_shape:
            raise ValueError(f"Text projector QPC expects {expected_shape}, got {tuple(hidden_states.shape)}")
        inputs = {"hidden_states": hidden_states}
        if self.has_attention_mask:
            if attention_mask is None:
                attention_mask = self._np.ones((1, self.sequence_length), dtype=self._np.bool_)
            else:
                attention_mask = self._np.asarray(
                    attention_mask.detach().cpu(),
                    dtype=self._binding_dtype("attention_mask", self._np.int8),
                )
            if tuple(attention_mask.shape) != (1, self.sequence_length):
                raise ValueError(
                    f"Text projector QPC expects attention_mask shape {(1, self.sequence_length)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            inputs["attention_mask"] = attention_mask
        output = self._run(inputs)
        return torch.from_numpy(output).float()


class LLaDAImageSigVQTokensAICSession:
    """Run the fixed-shape SigVQ token conditioning graph on AI100."""

    def __init__(self, qpc_path, *, sequence_length: int, feature_dim: int = 4096, device_ids=None):
        import numpy as np

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self._np = np
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
        self.session.set_buffers(
            {
                "semantic_features": np.zeros(
                    (1, self.sequence_length, self.feature_dim), dtype=self._binding_dtype("semantic_features", np.float32)
                ),
                "token_ids_out": np.zeros(
                    (1, self.sequence_length), dtype=self._binding_dtype("token_ids_out", np.int64)
                ),
            }
        )

    def _binding_dtype(self, name: str, default):
        index = self.session.binding_index_map.get(name)
        if index is None:
            return self._np.dtype(default)
        binding = self.session.bindings[index]
        return self.session.aic_to_np_dtype_mapping[binding.type]

    def run(self, token_ids):
        token_ids = self._np.asarray(
            token_ids.detach().cpu(), dtype=self._binding_dtype("token_ids", self._np.int64)
        )
        expected_shape = (1, self.sequence_length)
        if tuple(token_ids.shape) != expected_shape:
            raise ValueError(f"SigVQ token QPC expects {expected_shape}, got {tuple(token_ids.shape)}")
        output = self.session.run({"token_ids": token_ids})["semantic_features"]
        output = self._np.array(output, copy=True)
        expected_features = (1, self.sequence_length, self.feature_dim)
        if tuple(output.shape) != expected_features:
            raise RuntimeError(f"SigVQ token QPC returned {tuple(output.shape)}, expected {expected_features}")
        if not self._np.isfinite(output).all():
            raise RuntimeError("SigVQ token QPC returned non-finite semantic features")
        return torch.from_numpy(output).float()

    def close(self):
        if self.session is not None:
            self.session.deactivate()
            self.session = None


class LLaDAImageDenoisingAICSession:
    """Runtime adapter for one fixed-shape LLaDA-Image DiT QPC."""

    def __init__(
        self,
        qpc_path,
        *,
        mode: str,
        latent_shape: tuple[int, int, int, int],
        caption_length: int,
        semantic_length: int = 0,
        rope_config=None,
        device_ids=None,
        timings: dict[str, float] | None = None,
    ):
        import numpy as np

        from QEfficient.generation.cloud_infer import QAICInferenceSession

        if mode not in {"text", "vq", "editing"}:
            raise ValueError(f"Unsupported LLaDA-Image DiT mode: {mode}")
        self.mode = mode
        self.latent_shape = tuple(int(value) for value in latent_shape)
        self.caption_length = int(caption_length)
        self.semantic_length = int(semantic_length)
        self.rope_config = rope_config
        self._np = np
        self.session = QAICInferenceSession(str(qpc_path), device_ids=device_ids)
        self.dynamic_resolution = "image_rope_cos" in self.session.binding_index_map
        with _session_timing(timings, "component.dit_init.output_buffer_registration"):
            self.session.set_buffers({"sample": np.zeros(self.latent_shape, dtype=np.float32)})

    def run(
        self,
        hidden_states,
        timestep,
        cap_feats,
        cap_mask,
        semantic_feats=None,
        semantic_mask=None,
        source_latents=None,
    ):
        inputs = {
            "hidden_states": self._np.asarray(hidden_states.detach().cpu(), dtype=self._np.float32),
            "timestep": self._np.asarray(timestep.detach().cpu(), dtype=self._np.float32),
            "cap_feats": self._np.asarray(cap_feats.detach().cpu(), dtype=self._np.float32),
            "cap_mask": self._np.asarray(cap_mask.detach().cpu(), dtype=self._np.int64),
        }
        if self.mode in {"vq", "editing"}:
            if semantic_feats is None or semantic_mask is None:
                raise ValueError(f"{self.mode} DiT execution requires semantic features and a mask")
            inputs["semantic_feats"] = self._np.asarray(semantic_feats.detach().cpu(), dtype=self._np.float32)
            inputs["semantic_mask"] = self._np.asarray(semantic_mask.detach().cpu(), dtype=self._np.int64)
        if self.mode == "editing":
            if source_latents is None:
                raise ValueError("Editing DiT execution requires source latents")
            inputs["source_latents"] = self._np.asarray(source_latents.detach().cpu(), dtype=self._np.float32)
        if self.dynamic_resolution:
            if self.mode != "text" or self.rope_config is None:
                raise RuntimeError("Dynamic-resolution DiT QPC requires text mode and a transformer rope config")
            image_cos, image_sin, caption_cos, caption_sin = build_llada_image_text_rope(
                self.rope_config,
                hidden_states.shape[2],
                hidden_states.shape[3],
                self.caption_length,
            )
            inputs.update(
                {
                    "image_rope_cos": image_cos.numpy(),
                    "image_rope_sin": image_sin.numpy(),
                    "caption_rope_cos": caption_cos.numpy(),
                    "caption_rope_sin": caption_sin.numpy(),
                }
            )
        output = self.session.run(inputs)["sample"]
        output = self._np.array(output, copy=True)
        if tuple(output.shape) != self.latent_shape:
            raise RuntimeError(f"DiT QPC returned {tuple(output.shape)}, expected {self.latent_shape}")
        if not self._np.isfinite(output).all():
            raise RuntimeError("DiT QPC returned non-finite output")
        import torch

        return torch.from_numpy(output)

    def close(self):
        if self.session is not None:
            self.session.deactivate()
            self.session = None
