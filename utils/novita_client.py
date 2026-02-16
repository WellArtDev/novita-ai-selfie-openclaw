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
