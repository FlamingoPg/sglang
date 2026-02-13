# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.vaes.hunyuan_image import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

__all__ = [
    "HunyuanVAEConfig",
    "WanVAEConfig",
    "HunyuanImage3VAEConfig",
]
