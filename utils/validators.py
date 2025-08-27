"""
Input validation utilities.
"""

import re
from typing import Optional, Tuple
import base64
import io
from PIL import Image


def validate_prompt(prompt: str, max_length: int = 2000) -> Tuple[bool, Optional[str]]:
    """Validate text prompt."""
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    if len(prompt) > max_length:
        return False, f"Prompt exceeds maximum length of {max_length} characters"
    
    # Check for potentially harmful content patterns
    harmful_patterns = [
        r'\b(nude|naked|nsfw|adult|sexual|porn)\b',
        r'\b(violence|violent|blood|gore|kill|murder)\b',
        r'\b(hate|racist|nazi|terrorist)\b'
    ]
    
    prompt_lower = prompt.lower()
    for pattern in harmful_patterns:
        if re.search(pattern, prompt_lower):
            return False, "Prompt contains inappropriate content"
    
    return True, None


def validate_image_dimensions(width: int, height: int) -> Tuple[bool, Optional[str]]:
    """Validate image dimensions."""
    min_size = 64
    max_size = 2048
    
    if width < min_size or height < min_size:
        return False, f"Dimensions must be at least {min_size}x{min_size}"
    
    if width > max_size or height > max_size:
        return False, f"Dimensions must not exceed {max_size}x{max_size}"
    
    if width % 8 != 0 or height % 8 != 0:
        return False, "Dimensions must be divisible by 8"
    
    return True, None


def validate_base64_image(base64_string: str) -> Tuple[bool, Optional[str]]:
    """Validate base64 encoded image."""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Check base64 format
        try:
            base64.b64decode(base64_string, validate=True)
        except Exception:
            return False, "Invalid base64 encoding"
        
        # Try to decode and validate as image
        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify it's a valid image
            
            # Check image size
            image = Image.open(io.BytesIO(image_data))  # Reopen after verify()
            width, height = image.size
            
            is_valid, error = validate_image_dimensions(width, height)
            if not is_valid:
                return False, error
            
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"
        
        return True, None
        
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"


def validate_generation_parameters(
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """Validate generation parameters."""
    if not (10 <= num_inference_steps <= 100):
        return False, "num_inference_steps must be between 10 and 100"
    
    if not (1.0 <= guidance_scale <= 20.0):
        return False, "guidance_scale must be between 1.0 and 20.0"
    
    if seed is not None and seed < 0:
        return False, "seed must be non-negative"
    
    return True, None


def validate_editing_parameters(strength: float) -> Tuple[bool, Optional[str]]:
    """Validate image editing parameters."""
    if not (0.1 <= strength <= 1.0):
        return False, "strength must be between 0.1 and 1.0"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename.strip()