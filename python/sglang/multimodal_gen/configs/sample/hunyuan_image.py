# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 sampling parameters."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    """Sampling parameters for HunyuanImage-3.0 Base (T2I only)."""

    # Basic generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # Resolution (will be mapped to special tokens by tokenizer)
    # Supported resolutions: 1024x1024, 768x1280, 1280x768, etc. (37 aspect ratios)
    height: int = 1024
    width: int = 1024

    # Negative prompt
    negative_prompt: str = ""

    # Taylor Cache acceleration (HunyuanImage-specific optimization)
    use_taylor_cache: bool = False
    taylor_cache_interval: int = 5
    taylor_cache_order: int = 2
    taylor_cache_enable_first_enhance: bool = False

    # TeaCache (SGLang's standard acceleration)
    use_teacache: bool = False
    teacache_thresh: float = 0.3

    # System prompt control (for Instruct variants)
    use_system_prompt: bool = True
    system_prompt: str | None = None
    bot_task: str = "direct_gen"  # or "think_gen", "recaption", "unified"

    # MoE implementation
    moe_impl: str = "eager"  # or "flashinfer" for 3x speedup


@dataclass
class HunyuanImage3InstructSamplingParams(HunyuanImage3SamplingParams):
    """Sampling parameters for HunyuanImage-3.0-Instruct (full capabilities)."""

    # Instruct model benefits from slightly different defaults
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # Enable system prompt by default for Instruct
    use_system_prompt: bool = True
    bot_task: str = "direct_gen"


@dataclass
class HunyuanImage3InstructDistilSamplingParams(HunyuanImage3InstructSamplingParams):
    """Sampling parameters for HunyuanImage-3.0-Instruct-Distil (8-step fast)."""

    # Distilled model uses fewer steps
    num_inference_steps: int = 8
    guidance_scale: float = 7.5

    # Distilled models often benefit from Taylor Cache
    use_taylor_cache: bool = True
    taylor_cache_interval: int = 2
    taylor_cache_order: int = 2


@dataclass
class HunyuanImage3I2ISamplingParams(HunyuanImage3InstructSamplingParams):
    """Sampling parameters for HunyuanImage-3.0 Image-to-Image editing."""

    # I2I typically uses slightly fewer steps
    num_inference_steps: int = 40
    guidance_scale: float = 7.5

    # Denoising strength for I2I (0.0 = no change, 1.0 = full generation)
    strength: float = 0.8
