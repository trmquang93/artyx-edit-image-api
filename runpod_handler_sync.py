#!/usr/bin/env python3
"""
Synchronous RunPod serverless handler for AI image editing.
Completely avoids async/await to prevent event loop conflicts.
"""

import sys
import os
import time
import logging
import traceback
import base64
import io
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Synchronous image processing utilities."""
    
    def __init__(self):
        self.max_size = 2048
    
    def base64_to_pil(self, base64_string: str):
        """Convert base64 string to PIL Image."""
        try:
            from PIL import Image
            
            # Remove data URL prefix if present
            if base64_string.startswith('data:image/'):
                base64_string = base64_string.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Open as PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert base64 to PIL image: {e}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def pil_to_base64(self, image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format=format, quality=95, optimize=True)
            buffer.seek(0)
            
            # Encode to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"Failed to convert PIL image to base64: {e}")
            raise ValueError(f"Image conversion failed: {str(e)}")


class MockImageGenerator:
    """Mock image generator for testing without heavy ML models."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> str:
        """Generate mock image from text prompt."""
        try:
            logger.info(f"🎨 Generating {width}x{height} mock image: '{prompt}'")
            
            from PIL import Image, ImageDraw
            
            # Create gradient background
            image = Image.new('RGB', (width, height))
            draw = ImageDraw.Draw(image)
            
            # Create a colorful gradient based on prompt hash
            prompt_hash = hash(prompt) % 1000
            base_color = (
                (prompt_hash * 123) % 256,
                (prompt_hash * 456) % 256, 
                (prompt_hash * 789) % 256
            )
            
            # Draw gradient
            for y in range(height):
                factor = y / height
                color = (
                    int(base_color[0] * (1 - factor) + 255 * factor),
                    int(base_color[1] * (1 - factor) + 200 * factor),
                    int(base_color[2] * (1 - factor) + 150 * factor)
                )
                draw.line([(0, y), (width, y)], fill=color)
            
            # Add decorative elements
            import math
            center_x, center_y = width // 2, height // 2
            for i in range(5):
                radius = 20 + i * 15
                x1 = center_x - radius
                y1 = center_y - radius
                x2 = center_x + radius
                y2 = center_y + radius
                draw.ellipse([x1, y1, x2, y2], outline="white", width=2)
            
            # Add text
            try:
                prompt_text = prompt[:30] + "..." if len(prompt) > 30 else prompt
                draw.text((10, 10), f"Generated: {prompt_text}", fill="white")
                draw.text((10, height - 40), f"Size: {width}x{height}", fill="white")
                draw.text((10, height - 20), f"Steps: {num_inference_steps}", fill="white")
            except:
                # If text drawing fails, add a simple rectangle
                draw.rectangle([10, 10, 200, 50], fill="white")
            
            logger.info("✅ Mock image generation completed")
            
            # Convert to base64
            return self.image_processor.pil_to_base64(image)
            
        except Exception as e:
            logger.error(f"Mock image generation failed: {e}")
            raise
    
    def edit_image(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> str:
        """Edit image using text prompt (mock version)."""
        try:
            logger.info(f"🎨 Mock editing image with prompt: '{prompt}'")
            
            # Decode input image
            input_image = self.image_processor.base64_to_pil(image_base64)
            logger.info(f"✅ Input image decoded: {input_image.size}")
            
            from PIL import Image, ImageDraw, ImageFilter
            
            # Create edited version
            edited_image = input_image.copy()
            
            # Apply a simple filter effect
            if "blur" in prompt.lower():
                edited_image = edited_image.filter(ImageFilter.GaussianBlur(radius=2))
            elif "sharp" in prompt.lower():
                edited_image = edited_image.filter(ImageFilter.SHARPEN)
            elif "vintage" in prompt.lower() or "sepia" in prompt.lower():
                # Convert to sepia - simplified version
                try:
                    pixels = edited_image.load()
                    width, height = edited_image.size
                    for y in range(min(height, 100)):  # Limit processing for demo
                        for x in range(min(width, 100)):
                            try:
                                if pixels:
                                    pixel = pixels[x, y]
                                    if pixel and isinstance(pixel, (tuple, list)) and len(pixel) >= 3:
                                        r, g, b = pixel[:3]
                                        # Simple sepia effect
                                        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                                        pixels[x, y] = (min(255, gray + 40), min(255, gray + 20), gray)
                            except (IndexError, TypeError):
                                continue
                except Exception:
                    # If sepia fails, just continue with original
                    pass
            
            # Add visual indicator of editing
            draw = ImageDraw.Draw(edited_image)
            width, height = edited_image.size
            
            # Add colored border
            border_color = "red" if "sunset" in prompt.lower() else "blue"
            draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=3)
            
            # Add prompt text overlay
            try:
                prompt_text = prompt[:25] + "..." if len(prompt) > 25 else prompt
                # Add semi-transparent background for text
                draw.rectangle([5, 5, 250, 35], fill="black")
                draw.text((10, 10), f"Edit: {prompt_text}", fill="white")
            except:
                # Fallback: just add a colored rectangle
                draw.rectangle([5, 5, 100, 25], fill=border_color)
            
            logger.info("✅ Mock image editing completed")
            
            # Convert back to base64
            return self.image_processor.pil_to_base64(edited_image)
            
        except Exception as e:
            logger.error(f"Mock image editing failed: {e}")
            raise


# Global mock generator
mock_generator = None


def handler(job):
    """RunPod serverless handler function - completely synchronous."""
    logger.info("=== JOB RECEIVED ===")
    logger.info(f"Job input: {job}")
    
    try:
        global mock_generator
        
        # Initialize mock generator if not already done
        if mock_generator is None:
            logger.info("Initializing mock image generator...")
            mock_generator = MockImageGenerator()
            logger.info("✅ Mock generator initialized")
        
        job_input = job.get("input", {})
        task_type = job_input.get("task", "health")
        
        logger.info(f"Processing task: {task_type}")
        
        if task_type == "health":
            # Simple health check
            try:
                import torch
                torch_available = True
                gpu_available = torch.cuda.is_available()
            except ImportError:
                torch_available = False
                gpu_available = False
                
            return {
                "success": True,
                "status": "healthy",
                "message": "Sync AI Image Editing Server is running!",
                "environment": {
                    "python_version": sys.version.split()[0],
                    "timestamp": time.time(),
                    "server_type": "sync_ai_image_editing",
                    "torch_available": torch_available,
                    "gpu_available": gpu_available
                }
            }
            
        elif task_type == "generate":
            # Image generation
            prompt = job_input.get("prompt", "a beautiful landscape")
            negative_prompt = job_input.get("negative_prompt", "")
            width = job_input.get("width", 512)
            height = job_input.get("height", 512)
            num_inference_steps = job_input.get("num_inference_steps", 20)
            guidance_scale = job_input.get("guidance_scale", 7.5)
            seed = job_input.get("seed")
            
            start_time = time.time()
            result_image = mock_generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": f"Generated mock image for: {prompt}",
                "image": result_image,
                "metadata": {
                    "processing_time": processing_time,
                    "model": "MockGenerator",
                    "prompt": prompt,
                    "dimensions": f"{width}x{height}",
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                }
            }
            
        elif task_type == "edit" or task_type == "background_replacement":
            # Image editing
            image_base64 = job_input.get("image")
            image_url = job_input.get("image_url")
            prompt = job_input.get("prompt", "edit the image")
            negative_prompt = job_input.get("negative_prompt", "")
            num_inference_steps = job_input.get("num_inference_steps", 20)
            guidance_scale = job_input.get("guidance_scale", 7.5)
            strength = job_input.get("strength", 0.8)
            seed = job_input.get("seed")
            
            # Handle image input - either base64 or URL
            if image_url and not image_base64:
                try:
                    import requests
                    logger.info(f"Downloading image from URL: {image_url}")
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    
                    # Convert to base64
                    image_data = response.content
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    logger.info("✅ Image downloaded and converted to base64")
                    
                except Exception as e:
                    logger.error(f"Failed to download image from URL: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to download image: {str(e)}"
                    }
            
            if not image_base64:
                return {
                    "success": False,
                    "error": "No input image provided. Use 'image' (base64) or 'image_url'."
                }
            
            start_time = time.time()
            result_image = mock_generator.edit_image(
                image_base64=image_base64,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed
            )
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "message": f"Mock edited image with: {prompt}",
                "image": result_image,
                "metadata": {
                    "processing_time": processing_time,
                    "model": "MockEditor",
                    "prompt": prompt,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed
                }
            }
            
        elif task_type == "debug":
            # Debug info
            import subprocess
            
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                      capture_output=True, text=True, timeout=30)
                pip_list = result.stdout
            except:
                pip_list = "Could not get pip list"
            
            return {
                "success": True,
                "message": "Sync handler debug info",
                "python_version": sys.version,
                "installed_packages": pip_list[:1500],
                "environment_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "HOME": os.environ.get("HOME", ""),
                }
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}",
                "supported_tasks": ["health", "generate", "edit", "background_replacement", "debug"]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Handler exception: {str(e)}"
        }


def main():
    """Main entry point for RunPod serverless worker."""
    logger.info("🚀 Starting Sync AI Image Editing Server")
    logger.info(f"Python version: {sys.version}")
    
    try:
        import runpod
        logger.info("✅ RunPod imported successfully")
        
        logger.info("🎯 Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()