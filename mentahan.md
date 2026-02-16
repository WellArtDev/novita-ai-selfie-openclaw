# Novita.ai NSFW Image Generation Skill for OpenClaw

## Project Structure

```
novita-nsfw-skill/
├── skill.json
├── main.py
├── requirements.txt
├── handlers/
│   ├── __init__.py
│   └── image_generator.py
├── utils/
│   ├── __init__.py
│   ├── novita_client.py
│   └── prompt_builder.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── config/
│   ├── __init__.py
│   └── settings.py
└── README.md
```

## skill.json

```json
{
  "name": "novita_nsfw_image_generator",
  "version": "1.0.0",
  "description": "Generate NSFW images using Novita.ai API with various models and customization options",
  "author": "OpenClaw Community",
  "platform": "openclaw",
  "category": "image_generation",
  "tags": ["nsfw", "image", "ai", "novita", "generation", "adult"],
  "rating": "adult",
  "permissions": ["network", "storage"],
  "entry_point": "main.py",
  "config": {
    "novita_api_key": {
      "type": "string",
      "required": true,
      "description": "Your Novita.ai API key"
    },
    "default_model": {
      "type": "string",
      "default": "realisticVisionV51_v51VAE_97547.safetensors",
      "description": "Default Stable Diffusion model checkpoint"
    },
    "default_sampler": {
      "type": "string",
      "default": "DPM++ 2M Karras",
      "description": "Default sampling method"
    },
    "output_directory": {
      "type": "string",
      "default": "./outputs",
      "description": "Directory to save generated images"
    }
  },
  "commands": [
    {
      "name": "generate",
      "description": "Generate an NSFW image from a text prompt",
      "parameters": {
        "prompt": {"type": "string", "required": true},
        "negative_prompt": {"type": "string", "required": false},
        "width": {"type": "integer", "default": 512},
        "height": {"type": "integer", "default": 768},
        "steps": {"type": "integer", "default": 25},
        "cfg_scale": {"type": "number", "default": 7.0},
        "seed": {"type": "integer", "default": -1},
        "model": {"type": "string", "required": false},
        "sampler": {"type": "string", "required": false},
        "lora": {"type": "string", "required": false},
        "batch_size": {"type": "integer", "default": 1}
      }
    },
    {
      "name": "img2img",
      "description": "Generate NSFW image from an existing image + prompt",
      "parameters": {
        "image_url": {"type": "string", "required": true},
        "prompt": {"type": "string", "required": true},
        "negative_prompt": {"type": "string", "required": false},
        "strength": {"type": "number", "default": 0.7},
        "steps": {"type": "integer", "default": 25},
        "cfg_scale": {"type": "number", "default": 7.0},
        "seed": {"type": "integer", "default": -1}
      }
    },
    {
      "name": "list_models",
      "description": "List available NSFW-capable models"
    },
    {
      "name": "list_loras",
      "description": "List available LoRA models"
    },
    {
      "name": "upscale",
      "description": "Upscale a generated image",
      "parameters": {
        "image_url": {"type": "string", "required": true},
        "scale": {"type": "number", "default": 2.0}
      }
    }
  ]
}
```

## requirements.txt

```
requests>=2.31.0
pydantic>=2.0.0
Pillow>=10.0.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
```

## config/settings.py

```python
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
```

## models/schemas.py

```python
"""Pydantic models for request/response schemas."""

from typing import Optional, List
from pydantic import BaseModel, Field, validator


class Txt2ImgRequest(BaseModel):
    """Text-to-image generation request."""
    
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    width: int = Field(512, ge=128, le=2048, description="Image width")
    height: int = Field(768, ge=128, le=2048, description="Image height")
    steps: int = Field(25, ge=1, le=100, description="Number of sampling steps")
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0, description="CFG scale")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    model: Optional[str] = Field(None, description="Model checkpoint name")
    sampler: Optional[str] = Field(None, description="Sampler name")
    lora: Optional[str] = Field(None, description="LoRA model to apply")
    lora_weight: float = Field(0.8, ge=0.0, le=1.5, description="LoRA weight")
    batch_size: int = Field(1, ge=1, le=4, description="Number of images to generate")
    clip_skip: int = Field(2, ge=1, le=4, description="CLIP skip layers")
    
    @validator("width", "height")
    def must_be_multiple_of_8(cls, v):
        if v % 8 != 0:
            return (v // 8) * 8
        return v


class Img2ImgRequest(BaseModel):
    """Image-to-image generation request."""
    
    image_url: str = Field(..., description="Source image URL or base64")
    prompt: str = Field(..., description="Text prompt", min_length=1)
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    strength: float = Field(0.7, ge=0.0, le=1.0, description="Denoising strength")
    steps: int = Field(25, ge=1, le=100, description="Sampling steps")
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0, description="CFG scale")
    seed: int = Field(-1, description="Random seed")
    model: Optional[str] = Field(None, description="Model checkpoint")
    sampler: Optional[str] = Field(None, description="Sampler name")
    width: Optional[int] = Field(None, ge=128, le=2048)
    height: Optional[int] = Field(None, ge=128, le=2048)


class UpscaleRequest(BaseModel):
    """Image upscale request."""
    
    image_url: str = Field(..., description="Image URL to upscale")
    scale: float = Field(2.0, ge=1.0, le=4.0, description="Upscale factor")
    upscaler: str = Field("RealESRGAN_x4plus", description="Upscaler model")


class GeneratedImage(BaseModel):
    """A single generated image result."""
    
    image_url: str
    image_type: str = "png"
    seed: int
    width: int
    height: int
    nsfw_detected: bool = False


class GenerationResponse(BaseModel):
    """Response from image generation."""
    
    task_id: str
    status: str  # "QUEUED", "PROCESSING", "SUCCESS", "FAILED"
    images: List[GeneratedImage] = []
    model_used: str = ""
    generation_time: float = 0.0
    error_message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    """Task status polling response."""
    
    task_id: str
    status: str
    progress: float = 0.0
    eta_seconds: float = 0.0
    images: List[GeneratedImage] = []
```

## utils/novita_client.py

```python
"""Novita.ai API client wrapper."""

import time
import base64
import requests
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from config.settings import NovitaConfig
from models.schemas import (
    GeneratedImage,
    GenerationResponse,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)


class NovitaAPIError(Exception):
    """Custom exception for Novita API errors."""
    
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class NovitaClient:
    """Client for interacting with the Novita.ai API."""
    
    def __init__(self, config: NovitaConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })
    
    def _make_request(self, method: str, endpoint: str, payload: Optional[Dict] = None,
                      base_url: Optional[str] = None) -> Dict[str, Any]:
        """Make an API request to Novita."""
        url = f"{base_url or self.config.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.config.timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=payload, timeout=self.config.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code != 200:
                raise NovitaAPIError(
                    f"API request failed: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )
            
            return response.json()
        
        except requests.exceptions.Timeout:
            raise NovitaAPIError("Request timed out")
        except requests.exceptions.ConnectionError:
            raise NovitaAPIError("Connection error - check your network")
    
    def txt2img(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 768,
        steps: int = 25,
        cfg_scale: float = 7.0,
        seed: int = -1,
        model: Optional[str] = None,
        sampler: Optional[str] = None,
        lora: Optional[str] = None,
        lora_weight: float = 0.8,
        batch_size: int = 1,
        clip_skip: int = 2,
    ) -> GenerationResponse:
        """Generate images from text prompt using Novita.ai txt2img API."""
        
        model_name = model or self.config.default_model
        sampler_name = sampler or self.config.default_sampler
        neg_prompt = negative_prompt or self.config.default_negative_prompt
        
        # Build the API payload
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": cfg_scale,
            "seed": seed,
            "sampler_name": sampler_name,
            "batch_size": min(batch_size, self.config.max_batch_size),
            "clip_skip": clip_skip,
        }
        
        # Add LoRA if specified
        if lora:
            payload["loras"] = [
                {
                    "model_name": lora,
                    "strength": lora_weight,
                }
            ]
        
        logger.info(f"Submitting txt2img task: model={model_name}, size={width}x{height}")
        
        # Submit the task
        result = self._make_request(
            "POST",
            "async/txt2img",
            payload=payload,
        )
        
        task_id = result.get("task_id", "")
        
        if not task_id:
            raise NovitaAPIError("No task_id returned from API")
        
        # Poll for completion
        return self._poll_task(task_id)
    
    def img2img(
        self,
        image_url: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.7,
        steps: int = 25,
        cfg_scale: float = 7.0,
        seed: int = -1,
        model: Optional[str] = None,
        sampler: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> GenerationResponse:
        """Generate images from an input image + prompt."""
        
        model_name = model or self.config.default_model
        sampler_name = sampler or self.config.default_sampler
        neg_prompt = negative_prompt or self.config.default_negative_prompt
        
        # Determine if image_url is a URL or base64
        if image_url.startswith("http"):
            image_data = self._download_image_as_base64(image_url)
        else:
            image_data = image_url
        
        payload = {
            "model_name": model_name,
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "init_images": [image_data],
            "denoising_strength": strength,
            "steps": steps,
            "guidance_scale": cfg_scale,
            "seed": seed,
            "sampler_name": sampler_name,
        }
        
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height
        
        logger.info(f"Submitting img2img task: model={model_name}")
        
        result = self._make_request(
            "POST",
            "async/img2img",
            payload=payload,
        )
        
        task_id = result.get("task_id", "")
        if not task_id:
            raise NovitaAPIError("No task_id returned from API")
        
        return self._poll_task(task_id)
    
    def upscale(
        self,
        image_url: str,
        scale: float = 2.0,
        upscaler: str = "RealESRGAN_x4plus",
    ) -> GenerationResponse:
        """Upscale an image using Novita.ai upscaling API."""
        
        if image_url.startswith("http"):
            image_data = self._download_image_as_base64(image_url)
        else:
            image_data = image_url
        
        payload = {
            "image": image_data,
            "upscaling_resize": scale,
            "upscaler_1": upscaler,
        }
        
        logger.info(f"Submitting upscale task: scale={scale}x")
        
        result = self._make_request(
            "POST",
            "async/upscale",
            payload=payload,
        )
        
        task_id = result.get("task_id", "")
        if not task_id:
            raise NovitaAPIError("No task_id returned from API")
        
        return self._poll_task(task_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Novita.ai."""
        result = self._make_request(
            "GET",
            "models",
            base_url=self.config.base_url_v2,
        )
        
        models = result.get("data", {}).get("models", [])
        return models
    
    def list_loras(self) -> List[Dict[str, Any]]:
        """List available LoRA models from Novita.ai."""
        result = self._make_request(
            "GET",
            "loras",
            base_url=self.config.base_url_v2,
        )
        
        loras = result.get("data", {}).get("loras", [])
        return loras
    
    def _poll_task(
        self,
        task_id: str,
        max_wait: int = 300,
        poll_interval: float = 3.0,
    ) -> GenerationResponse:
        """Poll a task until completion."""
        
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            status_result = self._make_request(
                "GET",
                f"async/task-result?task_id={task_id}",
            )
            
            task = status_result.get("task", {})
            status = task.get("status", "UNKNOWN")
            
            logger.debug(f"Task {task_id}: status={status}")
            
            if status == "TASK_STATUS_SUCCEED":
                # Extract images
                images_data = task.get("images", [])
                images = []
                
                for idx, img_data in enumerate(images_data):
                    image_url = img_data.get("image_url", "")
                    image_type = img_data.get("image_type", "png")
                    
                    images.append(GeneratedImage(
                        image_url=image_url,
                        image_type=image_type,
                        seed=img_data.get("seed", -1),
                        width=img_data.get("width", 0),
                        height=img_data.get("height", 0),
                        nsfw_detected=img_data.get("nsfw_detection_result", False),
                    ))
                
                generation_time = time.time() - start_time
                
                return GenerationResponse(
                    task_id=task_id,
                    status="SUCCESS",
                    images=images,
                    model_used=task.get("model_name", ""),
                    generation_time=generation_time,
                )
            
            elif status == "TASK_STATUS_FAILED":
                error_msg = task.get("reason", "Unknown error")
                return GenerationResponse(
                    task_id=task_id,
                    status="FAILED",
                    error_message=error_msg,
                )
            
            elif status in ("TASK_STATUS_QUEUED", "TASK_STATUS_PROCESSING"):
                time.sleep(poll_interval)
            else:
                logger.warning(f"Unknown task status: {status}")
                time.sleep(poll_interval)
        
        return GenerationResponse(
            task_id=task_id,
            status="FAILED",
            error_message=f"Task timed out after {max_wait} seconds",
        )
    
    def _download_image_as_base64(self, url: str) -> str:
        """Download an image and convert to base64."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
```

## utils/prompt_builder.py

```python
"""Prompt building utilities for NSFW image generation."""

from typing import Optional, List, Dict


# Quality enhancement tokens
QUALITY_TAGS = {
    "high": "(masterpiece:1.2), (best quality:1.2), highres, ultra-detailed, sharp focus",
    "medium": "good quality, detailed",
    "photo": "(photorealistic:1.3), (RAW photo:1.2), 8k uhd, dslr, professional lighting, film grain",
    "anime": "(anime:1.2), detailed anime style, vibrant colors, clean lines",
    "artistic": "artistic, painterly, trending on artstation, concept art",
}

# Body type presets
BODY_PRESETS = {
    "slim": "slim body, slender figure, thin waist",
    "athletic": "athletic build, toned body, fit physique, muscular definition",
    "curvy": "curvy figure, voluptuous, hourglass figure, wide hips",
    "petite": "petite body, small frame, delicate features",
    "muscular": "muscular body, defined muscles, strong build",
    "thick": "thick body, full figured, plus size",
}

# Lighting presets
LIGHTING_PRESETS = {
    "studio": "studio lighting, softbox, professional lighting setup",
    "natural": "natural lighting, golden hour, warm sunlight",
    "dramatic": "dramatic lighting, rim light, chiaroscuro, high contrast",
    "neon": "neon lighting, cyberpunk atmosphere, colorful glow",
    "candlelight": "candlelight, warm ambient glow, intimate atmosphere",
    "moonlight": "moonlight, blue tones, night atmosphere, ethereal glow",
}

# Camera angle presets
CAMERA_PRESETS = {
    "portrait": "portrait shot, head and shoulders, face focus",
    "full_body": "full body shot, standing pose, full figure visible",
    "closeup": "extreme close-up, macro detail",
    "from_above": "from above, bird's eye view, looking down",
    "from_below": "from below, low angle, looking up",
    "side_view": "side view, profile shot",
    "three_quarter": "three-quarter view, slight angle",
    "pov": "POV shot, first person perspective",
}


class PromptBuilder:
    """Build optimized prompts for NSFW image generation."""
    
    def __init__(self):
        self._parts: List[str] = []
        self._quality: str = ""
        self._negative_parts: List[str] = []
    
    def reset(self) -> "PromptBuilder":
        """Reset the builder."""
        self._parts = []
        self._quality = ""
        self._negative_parts = []
        return self
    
    def set_quality(self, quality_level: str = "high") -> "PromptBuilder":
        """Set quality enhancement tags."""
        self._quality = QUALITY_TAGS.get(quality_level, QUALITY_TAGS["high"])
        return self
    
    def add_subject(self, description: str) -> "PromptBuilder":
        """Add the main subject description."""
        self._parts.append(description)
        return self
    
    def add_body_type(self, body_type: str) -> "PromptBuilder":
        """Add body type preset."""
        if body_type in BODY_PRESETS:
            self._parts.append(BODY_PRESETS[body_type])
        else:
            self._parts.append(body_type)
        return self
    
    def add_clothing(self, clothing: str) -> "PromptBuilder":
        """Add clothing/outfit description."""
        self._parts.append(clothing)
        return self
    
    def add_pose(self, pose: str) -> "PromptBuilder":
        """Add pose description."""
        self._parts.append(pose)
        return self
    
    def add_setting(self, setting: str) -> "PromptBuilder":
        """Add background/setting description."""
        self._parts.append(setting)
        return self
    
    def add_lighting(self, lighting: str) -> "PromptBuilder":
        """Add lighting preset or custom lighting."""
        if lighting in LIGHTING_PRESETS:
            self._parts.append(LIGHTING_PRESETS[lighting])
        else:
            self._parts.append(lighting)
        return self
    
    def add_camera(self, camera: str) -> "PromptBuilder":
        """Add camera angle preset or custom angle."""
        if camera in CAMERA_PRESETS:
            self._parts.append(CAMERA_PRESETS[camera])
        else:
            self._parts.append(camera)
        return self
    
    def add_custom(self, text: str) -> "PromptBuilder":
        """Add custom prompt text."""
        self._parts.append(text)
        return self
    
    def add_emphasis(self, text: str, weight: float = 1.3) -> "PromptBuilder":
        """Add emphasized text with weight."""
        self._parts.append(f"({text}:{weight})")
        return self
    
    def add_lora_trigger(self, trigger_word: str) -> "PromptBuilder":
        """Add a LoRA trigger word."""
        self._parts.append(f"<lora:{trigger_word}>")
        return self
    
    def add_negative(self, text: str) -> "PromptBuilder":
        """Add to negative prompt."""
        self._negative_parts.append(text)
        return self
    
    def build(self) -> str:
        """Build the final prompt string."""
        parts = []
        
        if self._quality:
            parts.append(self._quality)
        
        parts.extend(self._parts)
        
        return ", ".join(parts)
    
    def build_negative(self, include_defaults: bool = True) -> str:
        """Build the negative prompt string."""
        parts = []
        
        if include_defaults:
            parts.append(
                "(worst quality:1.4), (low quality:1.4), "
                "bad anatomy, bad hands, extra fingers, missing fingers, "
                "deformed, blurry, watermark, text, signature"
            )
        
        parts.extend(self._negative_parts)
        
        return ", ".join(parts)
    
    @staticmethod
    def enhance_prompt(prompt: str, style: str = "photo") -> str:
        """Quick enhance an existing prompt with quality tags."""
        quality = QUALITY_TAGS.get(style, QUALITY_TAGS["high"])
        return f"{quality}, {prompt}"
    
    @staticmethod
    def get_presets() -> Dict[str, Dict[str, str]]:
        """Get all available presets."""
        return {
            "quality": QUALITY_TAGS,
            "body_types": BODY_PRESETS,
            "lighting": LIGHTING_PRESETS,
            "camera": CAMERA_PRESETS,
        }
```

## handlers/image_generator.py

```python
"""Image generation handler for OpenClaw skill."""

import os
import logging
import base64
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

from config.settings import NovitaConfig, AVAILABLE_MODELS, AVAILABLE_SAMPLERS
from models.schemas import (
    Txt2ImgRequest,
    Img2ImgRequest,
    UpscaleRequest,
    GenerationResponse,
)
from utils.novita_client import NovitaClient, NovitaAPIError
from utils.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Main image generation handler."""
    
    def __init__(self, config: NovitaConfig):
        self.config = config
        self.client = NovitaClient(config)
        self.prompt_builder = PromptBuilder()
        
        # Ensure output directory exists
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
    
    async def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'generate' command - text-to-image generation."""
        
        try:
            request = Txt2ImgRequest(**params)
        except Exception as e:
            return self._error_response(f"Invalid parameters: {str(e)}")
        
        try:
            response = self.client.txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                seed=request.seed,
                model=request.model,
                sampler=request.sampler,
                lora=request.lora,
                lora_weight=request.lora_weight,
                batch_size=request.batch_size,
                clip_skip=request.clip_skip,
            )
            
            if response.status == "SUCCESS":
                # Save images locally
                saved_paths = self._save_images(response)
                
                return {
                    "success": True,
                    "task_id": response.task_id,
                    "images": [
                        {
                            "url": img.image_url,
                            "local_path": saved_paths[i] if i < len(saved_paths) else None,
                            "seed": img.seed,
                            "width": img.width,
                            "height": img.height,
                        }
                        for i, img in enumerate(response.images)
                    ],
                    "model_used": response.model_used,
                    "generation_time": round(response.generation_time, 2),
                    "message": f"Generated {len(response.images)} image(s) in {response.generation_time:.1f}s",
                }
            else:
                return self._error_response(
                    response.error_message or "Generation failed"
                )
        
        except NovitaAPIError as e:
            return self._error_response(f"API Error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error during generation")
            return self._error_response(f"Unexpected error: {str(e)}")
    
    async def img2img(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'img2img' command - image-to-image generation."""
        
        try:
            request = Img2ImgRequest(**params)
        except Exception as e:
            return self._error_response(f"Invalid parameters: {str(e)}")
        
        try:
            response = self.client.img2img(
                image_url=request.image_url,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                strength=request.strength,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                seed=request.seed,
                model=request.model,
                sampler=request.sampler,
                width=request.width,
                height=request.height,
            )
            
            if response.status == "SUCCESS":
                saved_paths = self._save_images(response)
                
                return {
                    "success": True,
                    "task_id": response.task_id,
                    "images": [
                        {
                            "url": img.image_url,
                            "local_path": saved_paths[i] if i < len(saved_paths) else None,
                            "seed": img.seed,
                            "width": img.width,
                            "height": img.height,
                        }
                        for i, img in enumerate(response.images)
                    ],
                    "model_used": response.model_used,
                    "generation_time": round(response.generation_time, 2),
                    "message": f"Img2img completed in {response.generation_time:.1f}s",
                }
            else:
                return self._error_response(
                    response.error_message or "Img2img failed"
                )
        
        except NovitaAPIError as e:
            return self._error_response(f"API Error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error during img2img")
            return self._error_response(f"Unexpected error: {str(e)}")
    
    async def upscale(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the 'upscale' command."""
        
        try:
            request = UpscaleRequest(**params)
        except Exception as e:
            return self._error_response(f"Invalid parameters: {str(e)}")
        
        try:
            response = self.client.upscale(
                image_url=request.image_url,
                scale=request.scale,
                upscaler=request.upscaler,
            )
            
            if response.status == "SUCCESS":
                saved_paths = self._save_images(response, prefix="upscaled")
                
                return {
                    "success": True,
                    "task_id": response.task_id,
                    "images": [
                        {
                            "url": img.image_url,
                            "local_path": saved_paths[i] if i < len(saved_paths) else None,
                            "width": img.width,
                            "height": img.height,
                        }
                        for i, img in enumerate(response.images)
                    ],
                    "generation_time": round(response.generation_time, 2),
                    "message": f"Upscaled {request.scale}x in {response.generation_time:.1f}s",
                }
            else:
                return self._error_response(
                    response.error_message or "Upscale failed"
                )
        
        except NovitaAPIError as e:
            return self._error_response(f"API Error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error during upscale")
            return self._error_response(f"Unexpected error: {str(e)}")
    
    async def list_models(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle the 'list_models' command."""
        
        try:
            # Return curated list + API models
            curated = [
                {"name": key, "filename": value}
                for key, value in AVAILABLE_MODELS.items()
            ]
            
            try:
                api_models = self.client.list_models()
            except Exception:
                api_models = []
            
            return {
                "success": True,
                "curated_models": curated,
                "api_models_count": len(api_models),
                "api_models": api_models[:50],  # Limit to first 50
                "available_samplers": AVAILABLE_SAMPLERS,
                "message": f"Found {len(curated)} curated models and {len(api_models)} API models",
            }
        
        except Exception as e:
            return self._error_response(f"Error listing models: {str(e)}")
    
    async def list_loras(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle the 'list_loras' command."""
        
        try:
            loras = self.client.list_loras()
            
            return {
                "success": True,
                "loras_count": len(loras),
                "loras": loras[:100],  # Limit to first 100
                "message": f"Found {len(loras)} LoRA models",
            }
        
        except Exception as e:
            return self._error_response(f"Error listing LoRAs: {str(e)}")
    
    def _save_images(
        self,
        response: GenerationResponse,
        prefix: str = "generated",
    ) -> List[str]:
        """Save generated images to disk."""
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, image in enumerate(response.images):
            try:
                # Download image
                img_response = requests.get(image.image_url, timeout=60)
                img_response.raise_for_status()
                
                # Generate filename
                filename = f"{prefix}_{timestamp}_{i}_{image.seed}.{image.image_type}"
                filepath = os.path.join(self.config.output_directory, filename)
                
                # Save to disk
                with open(filepath, "wb") as f:
                    f.write(img_response.content)
                
                saved_paths.append(filepath)
                logger.info(f"Saved image: {filepath}")
            
            except Exception as e:
                logger.error(f"Failed to save image {i}: {str(e)}")
                saved_paths.append(None)
        
        return saved_paths
    
    @staticmethod
    def _error_response(message: str) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "success": False,
            "error": message,
            "images": [],
        }
    
    def close(self):
        """Clean up resources."""
        self.client.close()
```

## main.py

```python
"""
Novita.ai NSFW Image Generation Skill for OpenClaw
Main entry point for the skill.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional

from config.settings import NovitaConfig
from handlers.image_generator import ImageGenerator
from utils.prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("novita_nsfw_skill")


class NovitaNSFWSkill:
    """
    OpenClaw skill for generating NSFW images using Novita.ai API.
    
    Supports:
    - Text-to-image (txt2img) generation
    - Image-to-image (img2img) generation
    - Image upscaling
    - Model and LoRA listing
    - Prompt building utilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the skill with configuration."""
        
        if config:
            self.config = NovitaConfig.from_skill_config(config)
        else:
            self.config = NovitaConfig.from_env()
        
        if not self.config.api_key:
            raise ValueError(
                "Novita.ai API key is required. Set NOVITA_API_KEY environment "
                "variable or provide it in skill configuration."
            )
        
        self.generator = ImageGenerator(self.config)
        self.prompt_builder = PromptBuilder()
        
        # Command routing table
        self._commands = {
            "generate": self.generator.generate,
            "img2img": self.generator.img2img,
            "upscale": self.generator.upscale,
            "list_models": self.generator.list_models,
            "list_loras": self.generator.list_loras,
            "build_prompt": self._handle_build_prompt,
            "get_presets": self._handle_get_presets,
        }
        
        logger.info("Novita NSFW Skill initialized successfully")
    
    async def handle(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main command handler - entry point for OpenClaw.
        
        Args:
            command: The command to execute
            params: Parameters for the command
            
        Returns:
            Response dictionary with results
        """
        
        params = params or {}
        
        if command not in self._commands:
            return {
                "success": False,
                "error": f"Unknown command: {command}",
                "available_commands": list(self._commands.keys()),
            }
        
        logger.info(f"Handling command: {command}")
        
        handler = self._commands[command]
        
        try:
            result = await handler(params)
            return result
        except Exception as e:
            logger.exception(f"Error handling command '{command}'")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _handle_build_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt building command."""
        
        builder = PromptBuilder()
        
        # Apply quality
        quality = params.get("quality", "high")
        builder.set_quality(quality)
        
        # Apply subject
        if "subject" in params:
            builder.add_subject(params["subject"])
        
        # Apply body type
        if "body_type" in params:
            builder.add_body_type(params["body_type"])
        
        # Apply clothing
        if "clothing" in params:
            builder.add_clothing(params["clothing"])
        
        # Apply pose
        if "pose" in params:
            builder.add_pose(params["pose"])
        
        # Apply setting
        if "setting" in params:
            builder.add_setting(params["setting"])
        
        # Apply lighting
        if "lighting" in params:
            builder.add_lighting(params["lighting"])
        
        # Apply camera
        if "camera" in params:
            builder.add_camera(params["camera"])
        
        # Custom additions
        if "custom" in params:
            if isinstance(params["custom"], list):
                for item in params["custom"]:
                    builder.add_custom(item)
            else:
                builder.add_custom(params["custom"])
        
        # Build prompts
        prompt = builder.build()
        negative_prompt = builder.build_negative(
            include_defaults=params.get("include_default_negatives", True)
        )
        
        return {
            "success": True,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
    
    async def _handle_get_presets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return all available presets."""
        return {
            "success": True,
            "presets": PromptBuilder.get_presets(),
        }
    
    def shutdown(self):
        """Clean up resources."""
        self.generator.close()
        logger.info("Novita NSFW Skill shut down")


# === OpenClaw Skill Interface Functions ===

_skill_instance: Optional[NovitaNSFWSkill] = None


def initialize(config: Dict[str, Any]) -> Dict[str, Any]:
    """Called by OpenClaw to initialize the skill."""
    global _skill_instance
    
    try:
        _skill_instance = NovitaNSFWSkill(config)
        return {
            "success": True,
            "message": "Novita NSFW Image Generator initialized",
            "version": "1.0.0",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def execute(command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Called by OpenClaw to execute a command."""
    global _skill_instance
    
    if _skill_instance is None:
        return {
            "success": False,
            "error": "Skill not initialized. Call initialize() first.",
        }
    
    # Run async handler in event loop
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            _skill_instance.handle(command, params)
        )
        return result
    finally:
        loop.close()


def shutdown():
    """Called by OpenClaw when shutting down the skill."""
    global _skill_instance
    
    if _skill_instance:
        _skill_instance.shutdown()
        _skill_instance = None


# === Standalone CLI Mode ===

def main():
    """CLI mode for testing the skill standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Novita.ai NSFW Image Generator")
    parser.add_argument("command", choices=[
        "generate", "img2img", "upscale", "list_models", "list_loras",
        "build_prompt", "get_presets"
    ])
    parser.add_argument("--prompt", "-p", type=str, help="Text prompt")
    parser.add_argument("--negative-prompt", "-n", type=str, help="Negative prompt")
    parser.add_argument("--width", "-W", type=int, default=512)
    parser.add_argument("--height", "-H", type=int, default=768)
    parser.add_argument("--steps", "-s", type=int, default=25)
    parser.add_argument("--cfg-scale", "-c", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--sampler", type=str, help="Sampler name")
    parser.add_argument("--lora", type=str, help="LoRA model")
    parser.add_argument("--lora-weight", type=float, default=0.8)
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--image-url", type=str, help="Input image URL (for img2img)")
    parser.add_argument("--strength", type=float, default=0.7, help="Denoising strength")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale factor")
    parser.add_argument("--output-dir", "-o", type=str, default="./outputs")
    parser.add_argument("--api-key", type=str, help="Novita API key")
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["NOVITA_API_KEY"] = args.api_key
    
    # Initialize
    config = {
        "novita_api_key": args.api_key or os.getenv("NOVITA_API_KEY", ""),
        "output_directory": args.output_dir,
    }
    
    if args.model:
        config["default_model"] = args.model
    
    init_result = initialize(config)
    
    if not init_result["success"]:
        print(f"❌ Initialization failed: {init_result['error']}")
        sys.exit(1)
    
    print(f"✅ {init_result['message']}")
    
    # Build params
    params = {}
    
    if args.command == "generate":
        if not args.prompt:
            print("❌ --prompt is required for generate command")
            sys.exit(1)
        
        params = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "model": args.model,
            "sampler": args.sampler,
            "lora": args.lora,
            "lora_weight": args.lora_weight,
            "batch_size": args.batch_size,
        }
    
    elif args.command == "img2img":
        if not args.prompt or not args.image_url:
            print("❌ --prompt and --image-url are required for img2img")
            sys.exit(1)
        
        params = {
            "image_url": args.image_url,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "strength": args.strength,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
        }
    
    elif args.command == "upscale":
        if not args.image_url:
            print("❌ --image-url is required for upscale")
            sys.exit(1)
        
        params = {
            "image_url": args.image_url,
            "scale": args.scale,
        }
    
    # Execute command
    print(f"\n🎨 Executing: {args.command}...")
    result = execute(args.command, params)
    
    # Display results
    print(f"\n{'='*60}")
    print(json.dumps(result, indent=2, default=str))
    
    if result.get("success"):
        images = result.get("images", [])
        if images:
            print(f"\n✅ Generated {len(images)} image(s):")
            for i, img in enumerate(images):
                print(f"  [{i+1}] URL: {img.get('url', 'N/A')}")
                if img.get("local_path"):
                    print(f"      Saved: {img['local_path']}")
                if img.get("seed"):
                    print(f"      Seed: {img['seed']}")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    # Cleanup
    shutdown()


if __name__ == "__main__":
    main()
```

## README.md

```markdown
# Novita.ai NSFW Image Generation Skill for OpenClaw

A comprehensive OpenClaw skill for generating NSFW images using the Novita.ai API.

## Features

- **Text-to-Image (txt2img)** - Generate images from text prompts
- **Image-to-Image (img2img)** - Transform existing images with prompts
- **Image Upscaling** - Upscale images up to 4x resolution
- **Model Selection** - Choose from dozens of Stable Diffusion models
- **LoRA Support** - Apply LoRA models for specific styles
- **Prompt Builder** - Built-in prompt construction utilities with presets
- **Batch Generation** - Generate multiple images at once
- **Auto-save** - Automatically saves generated images locally

## Setup

### 1. Get a Novita.ai API Key

1. Go to [novita.ai](https://novita.ai)
2. Create an account
3. Navigate to API Keys section
4. Generate a new API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Option A - Environment variable:
```bash
export NOVITA_API_KEY="your-api-key-here"
```

Option B - In OpenClaw skill config:
```json
{
  "novita_api_key": "your-api-key-here"
}
```

## Usage

### As OpenClaw Skill

```python
# The skill is automatically loaded by OpenClaw
# Use commands through the OpenClaw interface

# Generate an image
result = skill.execute("generate", {
    "prompt": "a beautiful woman, photorealistic, detailed skin",
    "width": 512,
    "height": 768,
    "steps": 25,
    "cfg_scale": 7.0,
    "model": "realisticVisionV51_v51VAE_97547.safetensors"
})

# Image-to-image
result = skill.execute("img2img", {
    "image_url": "https://example.com/input.jpg",
    "prompt": "enhance details, high quality",
    "strength": 0.6
})

# Upscale
result = skill.execute("upscale", {
    "image_url": "https://example.com/image.jpg",
    "scale": 2.0
})

# Build a prompt with presets
result = skill.execute("build_prompt", {
    "quality": "photo",
    "subject": "1girl, beautiful face, detailed eyes",
    "body_type": "athletic",
    "lighting": "studio",
    "camera": "portrait",
    "setting": "luxury bedroom"
})
```

### Standalone CLI

```bash
# Generate an image
python main.py generate \
  --prompt "1girl, beautiful, photorealistic, detailed" \
  --width 512 \
  --height 768 \
  --steps 30 \
  --model "realisticVisionV51_v51VAE_97547.safetensors"

# Image-to-image
python main.py img2img \
  --image-url "https://example.com/image.jpg" \
  --prompt "enhance, high quality" \
  --strength 0.7

# Upscale
python main.py upscale \
  --image-url "https://example.com/image.jpg" \
  --scale 2.0

# List available models
python main.py list_models

# List LoRA models
python main.py list_loras
```

## Prompt Builder Presets

### Quality Levels
- `high` - Masterpiece quality tags
- `medium` - Good quality tags
- `photo` - Photorealistic emphasis
- `anime` - Anime style emphasis
- `artistic` - Artistic/painterly style

### Body Types
- `slim`, `athletic`, `curvy`, `petite`, `muscular`, `thick`

### Lighting
- `studio`, `natural`, `dramatic`, `neon`, `candlelight`, `moonlight`

### Camera Angles
- `portrait`, `full_body`, `closeup`, `from_above`, `from_below`, `side_view`, `three_quarter`, `pov`

## Available Models

| Model | Filename | Best For |
|-------|----------|----------|
| Realistic Vision v5.1 | `realisticVisionV51_v51VAE_97547.safetensors` | Photorealistic |
| DreamShaper 8 | `dreamshaper_8_93211.safetensors` | Fantasy/Creative |
| MajicMix Realistic | `majicmixRealistic_v7_134792.safetensors` | Asian photorealistic |
| CyberRealistic | `cyberrealistic_v42_211145.safetensors` | Modern photorealistic |
| Absolute Reality | `absolutereality_v181_167592.safetensors` | Hyper-realistic |
| Perfect World | `perfectWorld_v6Baked_215507.safetensors` | Idealized realism |

## API Response Format

```json
{
  "success": true,
  "task_id": "abc-123",
  "images": [
    {
      "url": "https://cdn.novita.ai/...",
      "local_path": "./outputs/generated_20240101_120000_0_12345.png",
      "seed": 12345,
      "width": 512,
      "height": 768
    }
  ],
  "model_used": "realisticVisionV51_v51VAE_97547.safetensors",
  "generation_time": 8.5,
  "message": "Generated 1 image(s) in 8.5s"
}
```

## Error Handling

The skill handles common errors:
- Invalid API key
- Network timeouts
- Invalid parameters
- Generation failures
- Task polling timeouts

All errors return a consistent format:
```json
{
  "success": false,
  "error": "Descriptive error message",
  "images": []
}
```

## Notes

- Images are temporarily hosted on Novita.ai CDN (URLs expire)
- Always save images locally if you need to keep them
- Respect Novita.ai rate limits and terms of service
- API costs apply per generation - check novita.ai pricing
```

## `__init__.py` Files

### `config/__init__.py`
```python
from .settings import NovitaConfig, AVAILABLE_MODELS, AVAILABLE_SAMPLERS
```

### `models/__init__.py`
```python
from .schemas import (
    Txt2ImgRequest,
    Img2ImgRequest,
    UpscaleRequest,
    GeneratedImage,
    GenerationResponse,
    TaskStatusResponse,
)
```

### `utils/__init__.py`
```python
from .novita_client import NovitaClient, NovitaAPIError
from .prompt_builder import PromptBuilder
```

### `handlers/__init__.py`
```python
from .image_generator import ImageGenerator
```

---

## Quick Start Example

```python
import os
os.environ["NOVITA_API_KEY"] = "your-key-here"

from main import initialize, execute, shutdown

# Initialize
initialize({"novita_api_key": os.environ["NOVITA_API_KEY"]})

# Generate
result = execute("generate", {
    "prompt": "(masterpiece:1.2), (best quality:1.2), 1girl, beautiful detailed face, photorealistic",
    "negative_prompt": "(worst quality:1.4), bad anatomy, blurry",
    "width": 512,
    "height": 768,
    "steps": 28,
    "cfg_scale": 7.0,
    "seed": -1,
})

print(result)

# Cleanup
shutdown()
```

This skill provides a complete, production-ready integration with the Novita.ai API for NSFW image generation within the OpenClaw framework. It includes proper error handling, async support, image saving, prompt building utilities, and both programmatic and CLI interfaces.
