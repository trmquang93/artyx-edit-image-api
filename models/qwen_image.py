"""
Qwen-Image model manager for text-to-image generation and image editing.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional, Dict, Any, Union
import threading

import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import set_seed

from models.image_processor import ImageProcessor


logger = logging.getLogger(__name__)


class QwenImageManager:
    """Manages Qwen-Image models for text-to-image and image editing."""
    
    def __init__(self):
        self.text_to_image_pipeline = None
        self.image_edit_pipeline = None
        self.image_processor = ImageProcessor()
        self.device = None
        self.torch_dtype = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # Model configurations - Pure Qwen models only
        self.text_to_image_model = "Qwen/Qwen-Image"
        self.image_edit_model = "Qwen/Qwen-Image-Edit"
        
        # Magic prompts for enhanced quality
        self.positive_magic = {
            "en": ", masterpiece, best quality, detailed, realistic, high resolution",
            "zh": "，杰作，最佳质量，详细，逼真，高分辨率"
        }
    
    async def initialize(self):
        """Initialize the model pipelines."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            logger.info("Initializing Qwen-Image models...")
            
            # Set device and dtype
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
            
            # Initialize text-to-image pipeline
            await self._initialize_text_to_image()
            
            # Initialize image editing pipeline
            await self._initialize_image_edit()
            
            self._initialized = True
            logger.info("Qwen-Image models initialized successfully")
    
    async def _initialize_text_to_image(self):
        """Initialize the Qwen-Image text-to-image pipeline."""
        try:
            logger.info(f"Loading Qwen text-to-image model: {self.text_to_image_model}")
            
            self.text_to_image_pipeline = DiffusionPipeline.from_pretrained(
                self.text_to_image_model,  # "Qwen/Qwen-Image"
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            self.text_to_image_pipeline = self.text_to_image_pipeline.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.text_to_image_pipeline, 'enable_attention_slicing'):
                self.text_to_image_pipeline.enable_attention_slicing()
            
            if hasattr(self.text_to_image_pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.text_to_image_pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
            
            logger.info("Qwen text-to-image pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen text-to-image pipeline: {e}")
            raise RuntimeError(f"Qwen-Image model could not be loaded: {str(e)}")
    
    async def _initialize_image_edit(self):
        """Initialize the Qwen-Image-Edit pipeline."""
        try:
            logger.info(f"Loading Qwen image editing model: {self.image_edit_model}")
            
            self.image_edit_pipeline = DiffusionPipeline.from_pretrained(
                self.image_edit_model,  # "Qwen/Qwen-Image-Edit"
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            self.image_edit_pipeline = self.image_edit_pipeline.to(self.device)
            
            # Enable memory optimizations
            if hasattr(self.image_edit_pipeline, 'enable_attention_slicing'):
                self.image_edit_pipeline.enable_attention_slicing()
            
            if hasattr(self.image_edit_pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.image_edit_pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers for Qwen editing: {e}")
            
            logger.info("Qwen image editing pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen image editing pipeline: {e}")
            raise RuntimeError(f"Qwen-Image-Edit model could not be loaded: {str(e)}")
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: Optional[int] = None
    ) -> str:
        """Generate image from text prompt."""
        if not self._initialized:
            await self.initialize()
        
        if not self.text_to_image_pipeline:
            raise RuntimeError("Text-to-image pipeline not available")
        
        try:
            # Set seed for reproducibility
            if seed is not None:
                set_seed(seed)
            
            # Enhance prompt with magic words
            enhanced_prompt = prompt + self.positive_magic["en"]
            
            logger.info(f"Generating {width}x{height} image with {num_inference_steps} steps")
            
            # Generate image
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_sync,
                enhanced_prompt,
                negative_prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale
            )
            
            # Convert to base64
            return await self.image_processor.pil_to_base64(result.images[0])
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def _generate_sync(self, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale):
        """Synchronous image generation for executor."""
        if self.text_to_image_pipeline is None:
            raise RuntimeError("Text-to-image pipeline not available")
        return self.text_to_image_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    
    async def edit_image(
        self,
        image_base64: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        strength: float = 0.8,
        seed: Optional[int] = None
    ) -> str:
        """Edit image using text prompt - background replacement focus."""
        if not self._initialized:
            await self.initialize()
        
        if not self.image_edit_pipeline:
            raise RuntimeError("Image editing pipeline not available")
        
        try:
            # Set seed for reproducibility
            if seed is not None:
                set_seed(seed)
            
            # Decode input image
            input_image = await self.image_processor.base64_to_pil(image_base64)
            
            # Create a basic mask for background replacement
            # For now, create a simple mask that covers background areas
            mask_image = await self._create_background_mask(input_image)
            
            logger.info(f"Editing image with {num_inference_steps} steps, strength {strength}")
            
            # Enhance prompt for background replacement
            enhanced_prompt = f"Replace background with: {prompt}, high quality, detailed"
            if negative_prompt is None:
                negative_prompt = "blurry, low quality, distorted, bad anatomy"
            
            # Edit image using inpainting
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._edit_sync,
                input_image,
                mask_image,
                enhanced_prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                strength
            )
            
            # Convert to base64
            return await self.image_processor.pil_to_base64(result.images[0])
            
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
    
    def _edit_sync(self, image, mask_image, prompt, negative_prompt, num_inference_steps, guidance_scale, strength):
        """Synchronous image editing for executor."""
        if self.image_edit_pipeline is None:
            raise RuntimeError("Image editing pipeline not available")
        return self.image_edit_pipeline(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        )
    
    async def _create_background_mask(self, image):
        """Create a simple background mask for inpainting."""
        # For now, create a basic mask that targets background areas
        # In production, this could use segmentation models for better masks
        from PIL import Image as PILImage, ImageDraw
        
        # Create a mask that covers outer edges (typical background areas)
        width, height = image.size
        mask = PILImage.new('L', (width, height), 0)  # Black mask
        draw = ImageDraw.Draw(mask)
        
        # Create a border mask (white = inpaint, black = keep)
        border_width = min(width, height) // 8
        draw.rectangle([0, 0, width, border_width], fill=255)  # Top
        draw.rectangle([0, height-border_width, width, height], fill=255)  # Bottom
        draw.rectangle([0, 0, border_width, height], fill=255)  # Left
        draw.rectangle([width-border_width, 0, width, height], fill=255)  # Right
        
        # Add some center areas for more natural background replacement
        center_x, center_y = width // 2, height // 2
        ellipse_w, ellipse_h = width // 3, height // 3
        draw.ellipse([
            center_x - ellipse_w, center_y - ellipse_h,
            center_x + ellipse_w, center_y + ellipse_h
        ], fill=0)  # Keep center subject
        
        return mask
    
    async def get_health_info(self) -> Dict[str, Any]:
        """Get health information about the model manager."""
        health = {
            "model_loaded": self._initialized,
            "gpu_available": torch.cuda.is_available(),
        }
        
        # Add GPU memory info if available
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                health["memory_usage"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "utilization": round(memory_allocated / memory_total * 100, 1)
                }
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return health
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "model_name": "Qwen-Image",
            "model_type": "text-to-image, image-editing",
            "capabilities": [
                "text-to-image generation",
                "image editing",
                "multi-language support",
                "high-resolution output"
            ],
            "supported_formats": ["PNG", "JPEG", "WebP"],
            "max_resolution": {"width": 2048, "height": 2048}
        }
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        status = {
            "initialized": self._initialized,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "text_to_image_available": self.text_to_image_pipeline is not None,
            "image_edit_available": self.image_edit_pipeline is not None,
        }
        
        # Add model info
        health = await self.get_health_info()
        status.update(health)
        
        return status
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up Qwen-Image models...")
        
        with self._lock:
            if self.text_to_image_pipeline:
                del self.text_to_image_pipeline
                self.text_to_image_pipeline = None
            
            if self.image_edit_pipeline:
                del self.image_edit_pipeline
                self.image_edit_pipeline = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._initialized = False
        
        logger.info("Cleanup completed")