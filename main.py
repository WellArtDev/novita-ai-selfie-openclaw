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
