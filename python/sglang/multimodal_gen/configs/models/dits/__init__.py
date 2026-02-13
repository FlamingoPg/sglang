# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.dits.hunyuan_image import (
    HunyuanImage3DiTConfig,
    HunyuanImage3InstructDiTConfig,
    HunyuanImage3InstructDistilDiTConfig,
)
from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import HunyuanVideoConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig

__all__ = [
    "HunyuanVideoConfig",
    "WanVideoConfig",
    "HunyuanImage3DiTConfig",
    "HunyuanImage3InstructDiTConfig",
    "HunyuanImage3InstructDistilDiTConfig",
]
