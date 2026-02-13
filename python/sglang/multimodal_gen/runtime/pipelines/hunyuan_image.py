# SPDX-License-Identifier: Apache-2.0
"""
HunyuanImage-3.0 diffusion pipeline implementation.

This module contains the implementation of the HunyuanImage-3.0 pipeline
using the modular pipeline architecture from SGLang.
"""

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    """Pipeline for HunyuanImage-3.0 Base (T2I only)."""

    pipeline_name = "HunyuanImage3Pipeline"

    _required_config_modules = [
        "tokenizer",
        "text_encoder",  # HunyuanImage uses the main model as text encoder
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""

        # Stage 1: Input validation
        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage()
        )

        # Stage 2: Text encoding
        # HunyuanImage-3.0 uses its own transformer for text encoding
        # The tokenizer handles special tokens for image size/ratio
        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        # Stage 3: Conditioning (prepare embeddings for diffusion)
        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage()
        )

        # Stage 4: Timestep preparation
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")
            ),
        )

        # Stage 5: Latent preparation (initialize noise)
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Stage 6: Denoising (main diffusion loop)
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Stage 7: VAE decoding
        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae"))
        )


class HunyuanImage3InstructPipeline(HunyuanImage3Pipeline):
    """Pipeline for HunyuanImage-3.0-Instruct (full capabilities: T2I, I2I, TI2I)."""

    pipeline_name = "HunyuanImage3InstructPipeline"

    # Same stages as base, but Instruct model can handle I2I tasks
    # Additional stages for image encoding will be added if needed


class HunyuanImage3InstructDistilPipeline(HunyuanImage3InstructPipeline):
    """Pipeline for HunyuanImage-3.0-Instruct-Distil (8-step fast inference)."""

    pipeline_name = "HunyuanImage3InstructDistilPipeline"

    # Same stages, but uses distilled weights and fewer inference steps


# Export all pipelines for automatic discovery
EntryClass = [
    HunyuanImage3Pipeline,
    HunyuanImage3InstructPipeline,
    HunyuanImage3InstructDistilPipeline,
]
