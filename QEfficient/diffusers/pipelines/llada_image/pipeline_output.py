# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass

from diffusers.utils import BaseOutput


@dataclass
class LLaDAImagePipelineOutput(BaseOutput):
    images: object
