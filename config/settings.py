"""Configuration settings for Novita NSFW Skill."""

import os
from dataclasses import dataclass, field
from typing import Optional


NOVITA_API_BASE = "https://api.novita.ai/v3"
NOVITA_API_V2_BASE = "https://api.novita.ai/v2"

# Popular NSFW-capable models on Novita.ai
AVAILABLE_MODELS = {
    "realistic_vision_v51": "realisticVisionV51_v51VAE_97547.safetensors",
    "dreamshaper_8": "dreamshaper_8_93211.safetensors",
    "majicmix_realistic": "majicmixRealistic_v7_134792.safetensors",
    "cyberrealistic": "cyberrealistic_v42_211145.safetensors",
    "absolute_reality": "absolutereality_v181_167592.safetensors",
    "perfect_world": "perfectWorld_v6Baked_215507.safetensors",
    "bb95_furry_mix": "bb95FurryMix_v100_144309.safetensors",
    "unstable_diffusers": "unstableDiffusers_v11_245559.safetensors",
    "epinikion": "epinikion_v2_222460.safetensors",
    "hassanblend": "hassanblend_v15_139866.safetensors",
}

AVAILABLE_SAMPLERS = [
    "Euler",
    "Euler a",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DPM++ 2M SDE Karras",
    "DDIM",
    "LMS",
    "UniPC",
    "DPM2 Karras",
    "Heun",
]

DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality:1.4), (low quality:1.4), (monochrome:1.1), "
    "bad anatomy, bad hands, extra fingers, missing fingers, "
    "deformed, blurry, watermark, text, signature, "
    "poorly drawn face, mutation, extra limbs"
)


@dataclass
class NovitaConfig:
    """Novita API configuration."""
    
    api_key: str = ""
    base_url: str = NOVITA_API_BASE
    base_url_v2: str = NOVITA_API_V2_BASE
    default_model: str = AVAILABLE_MODELS["realistic_vision_v51"]
    default_sampler: str = "DPM++ 2M Karras"
    default_width: int = 512
    default_height: int = 768
    default_steps: int = 25
    default_cfg_scale: float = 7.0
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_directory: str = "./outputs"
    max_batch_size: int = 4
    timeout: int = 120
    
    @classmethod
    def from_env(cls) -> "NovitaConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("NOVITA_API_KEY", ""),
            default_model=os.getenv("NOVITA_DEFAULT_MODEL", AVAILABLE_MODELS["realistic_vision_v51"]),
            default_sampler=os.getenv("NOVITA_DEFAULT_SAMPLER", "DPM++ 2M Karras"),
            output_directory=os.getenv("NOVITA_OUTPUT_DIR", "./outputs"),
        )
    
    @classmethod
    def from_skill_config(cls, config: dict) -> "NovitaConfig":
        """Create config from OpenClaw skill configuration."""
        return cls(
            api_key=config.get("novita_api_key", os.getenv("NOVITA_API_KEY", "")),
            default_model=config.get("default_model", AVAILABLE_MODELS["realistic_vision_v51"]),
            default_sampler=config.get("default_sampler", "DPM++ 2M Karras"),
            output_directory=config.get("output_directory", "./outputs"),
        )
