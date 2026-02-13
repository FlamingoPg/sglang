# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 DiT configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class HunyuanImage3ArchConfig(DiTArchConfig):
    """Architecture configuration for HunyuanImage-3.0 80B MoE DiT."""

    # Model architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    attention_head_dim: int = 128  # hidden_size / num_attention_heads
    num_hidden_layers: int = 48
    intermediate_size: int = 11008

    # MoE specific
    num_experts: int = 64
    num_shared_expert: int = 1
    moe_topk: int = 6  # 13B activated (64 experts, topk=6 + 1 shared)
    moe_intermediate_size: int = 2752

    # Latent configuration
    patch_size: int = 2
    in_channels: int = 16  # VAE latent channels * 4 (packed)
    out_channels: int = 16
    num_channels_latents: int = 4  # Before packing

    # Attention settings
    use_qk_norm: bool = True
    attention_dropout: float = 0.0

    # RoPE settings
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048

    # Normalization
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels


@dataclass
class HunyuanImage3DiTConfig(DiTConfig):
    """DiT configuration for HunyuanImage-3.0."""

    arch_config: DiTArchConfig = field(default_factory=HunyuanImage3ArchConfig)
    prefix: str = "hunyuan_image_3"


@dataclass
class HunyuanImage3InstructDiTConfig(HunyuanImage3DiTConfig):
    """DiT configuration for HunyuanImage-3.0-Instruct variant."""

    prefix: str = "hunyuan_image_3_instruct"


@dataclass
class HunyuanImage3InstructDistilDiTConfig(HunyuanImage3DiTConfig):
    """DiT configuration for HunyuanImage-3.0-Instruct-Distil variant (8-step fast)."""

    prefix: str = "hunyuan_image_3_instruct_distil"
