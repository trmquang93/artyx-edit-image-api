"""
Pydantic schemas for API request/response models.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import base64
import io
from PIL import Image


class GenerateImageRequest(BaseModel):
    """Request model for text-to-image generation."""
    
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="Negative prompt to avoid certain elements")
    width: int = Field(1024, ge=256, le=2048, description="Image width (must be divisible by 8)")
    height: int = Field(1024, ge=256, le=2048, description="Image height (must be divisible by 8)")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(4.0, ge=1.0, le=20.0, description="Guidance scale for classifier-free guidance")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducible results")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v % 8 != 0:
            raise ValueError("Width and height must be divisible by 8")
        return v


class EditImageRequest(BaseModel):
    """Request model for image editing."""
    
    image: str = Field(..., description="Base64 encoded input image")
    prompt: str = Field(..., min_length=1, max_length=2000, description="Edit instruction prompt")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="Negative prompt")
    num_inference_steps: int = Field(50, ge=10, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(4.0, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, ge=0, description="Random seed")
    strength: float = Field(0.8, ge=0.1, le=1.0, description="Editing strength (0.1=subtle, 1.0=strong)")
    
    @validator('image')
    def validate_base64_image(cls, v):
        """Validate that the image is valid base64 and can be decoded."""
        try:
            # Remove data URL prefix if present
            if v.startswith('data:image/'):
                v = v.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(v)
            
            # Verify it's a valid image
            image = Image.open(io.BytesIO(image_data))
            image.verify()
            
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")


class ImageResponse(BaseModel):
    """Response model for image generation/editing."""
    
    success: bool = Field(..., description="Whether the operation was successful")
    image: Optional[str] = Field(None, description="Base64 encoded result image")
    message: str = Field(..., description="Status message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Health check message")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of model (text-to-image, image-editing)")
    capabilities: List[str] = Field(..., description="List of model capabilities")
    supported_formats: List[str] = Field(..., description="Supported image formats")
    max_resolution: Dict[str, int] = Field(..., description="Maximum supported resolution")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")