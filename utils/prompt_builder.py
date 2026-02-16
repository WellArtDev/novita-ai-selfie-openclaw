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
