# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 pipeline configuration."""

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image import (
    HunyuanImage3DiTConfig,
    HunyuanImage3InstructDiTConfig,
    HunyuanImage3InstructDistilDiTConfig,
)
from sglang.multimodal_gen.configs.models.vaes.hunyuan_image import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def hunyuan_image_preprocess_text(prompt: str) -> str:
    """Preprocess text for HunyuanImage-3.0 tokenizer."""
    # The tokenizer handles prompt templates internally via system prompts
    return prompt


def hunyuan_image_postprocess_text(outputs, _text_inputs) -> torch.Tensor:
    """Postprocess text encoder outputs for HunyuanImage-3.0."""
    # Use the last hidden state from the transformer
    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
    return hidden_states


@dataclass
class HunyuanImage3PipelineConfig(ImagePipelineConfig):
    """Configuration for HunyuanImage-3.0 T2I pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2I

    # Model components
    dit_config: DiTConfig = field(default_factory=HunyuanImage3DiTConfig)
    vae_config: VAEConfig = field(default_factory=HunyuanImage3VAEConfig)

    # Text encoder configuration
    # HunyuanImage-3.0 uses its own transformer model as text encoder
    # The tokenizer is part of the main model
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: ()  # Will use the DiT's built-in text encoder
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (hunyuan_image_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (hunyuan_image_postprocess_text,)
    )

    # Diffusion parameters
    embedded_cfg_scale: float = 7.5
    flow_shift: float = 1.0  # FlowMatch shift parameter

    # Precision settings
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    vae_autocast_precision: str = "fp32"  # For VAE decode stability

    # MoE settings
    moe_impl: str = "eager"  # or "flashinfer" for 3x speedup
    moe_drop_tokens: bool = True

    # Special features
    use_taylor_cache: bool = False
    taylor_cache_interval: int = 5
    taylor_cache_order: int = 2

    # Guidance settings
    should_use_guidance: bool = True

    def __post_init__(self):
        # T2I mode: only need decoder
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class HunyuanImage3InstructPipelineConfig(HunyuanImage3PipelineConfig):
    """Configuration for HunyuanImage-3.0-Instruct (full capabilities: T2I, I2I, TI2I)."""

    task_type: ModelTaskType = ModelTaskType.T2I  # Can also do I2I, TI2I

    dit_config: DiTConfig = field(default_factory=HunyuanImage3InstructDiTConfig)

    def __post_init__(self):
        # Instruct mode: need both encoder and decoder for I2I tasks
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class HunyuanImage3InstructDistilPipelineConfig(HunyuanImage3InstructPipelineConfig):
    """Configuration for HunyuanImage-3.0-Instruct-Distil (8-step fast inference)."""

    dit_config: DiTConfig = field(default_factory=HunyuanImage3InstructDistilDiTConfig)

    # Distilled model uses fewer inference steps by default
    embedded_cfg_scale: float = 7.5
    flow_shift: float = 1.0


@dataclass
class HunyuanImage3I2IPipelineConfig(HunyuanImage3InstructPipelineConfig):
    """Configuration for HunyuanImage-3.0 Image-to-Image editing."""

    task_type: ModelTaskType = ModelTaskType.I2I

    def __post_init__(self):
        super().__post_init__()
        # I2I requires both VAE encoder and decoder
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
