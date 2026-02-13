# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 VAE configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanImage3VAEArchConfig(VAEArchConfig):
    """Architecture configuration for HunyuanImage-3.0 Conv3D VAE."""

    # VAE architecture
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    # Scaling factors
    scaling_factor: float = 0.13025
    shift_factor: float = 0.0

    # Compression
    spatial_compression_ratio: int = 8  # 8x8 downsampling
    temporal_compression_ratio: int = 1  # No temporal compression (image model)

    # Architecture details
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2

    def __post_init__(self):
        super().__post_init__()
        # Set vae_scale_factor for compatibility
        self.vae_scale_factor = self.spatial_compression_ratio


@dataclass
class HunyuanImage3VAEConfig(VAEConfig):
    """VAE configuration for HunyuanImage-3.0."""

    arch_config: VAEArchConfig = field(default_factory=HunyuanImage3VAEArchConfig)

    # Load settings
    load_encoder: bool = True  # For I2I tasks
    load_decoder: bool = True

    # Tiling settings (for large images)
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    # Distributed VAE support
    use_dist_vae: bool = False

    def get_vae_scale_factor(self):
        return self.arch_config.spatial_compression_ratio

    def __post_init__(self):
        # HunyuanImage VAE doesn't use temporal tiling
        self.blend_num_frames = 0

    def post_init(self):
        self.arch_config.vae_scale_factor = self.arch_config.spatial_compression_ratio
