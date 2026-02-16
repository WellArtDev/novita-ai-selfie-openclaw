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
