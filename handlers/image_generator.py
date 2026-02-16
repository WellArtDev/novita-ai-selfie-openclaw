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
