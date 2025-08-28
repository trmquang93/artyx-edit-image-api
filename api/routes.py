"""
FastAPI routes for AI image editing endpoints.
"""

import asyncio
import logging
import time
import base64
import io
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image

from api.schemas import (
    GenerateImageRequest,
    EditImageRequest,
    ImageResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
    MultipartEditRequest,
    MultipartBackgroundRequest,
    FileUploadResponse
)
from models.qwen_image import QwenImageManager


router = APIRouter()
logger = logging.getLogger(__name__)


def get_model_manager(request: Request) -> QwenImageManager:
    """Dependency to get model manager from app state."""
    model_manager = getattr(request.app.state, 'model_manager', None)
    if not model_manager:
        raise HTTPException(
            status_code=503,
            detail="Model manager not initialized"
        )
    return model_manager


@router.get("/health", response_model=HealthResponse)
async def health_check(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Health check endpoint."""
    try:
        health_info = await model_manager.get_health_info()
        
        return HealthResponse(
            status="healthy",
            message="Service is running normally",
            model_loaded=health_info["model_loaded"],
            gpu_available=health_info["gpu_available"],
            memory_usage=health_info.get("memory_usage")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            model_loaded=False,
            gpu_available=False,
            memory_usage=None
        )


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Get information about loaded models."""
    try:
        info = await model_manager.get_model_info()
        
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info["model_type"],
            capabilities=info["capabilities"],
            supported_formats=info["supported_formats"],
            max_resolution=info["max_resolution"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.post("/generate", response_model=ImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Generate image from text prompt using Qwen-Image."""
    start_time = time.time()
    
    try:
        logger.info(f"Generating image with prompt: {request.prompt[:100]}...")
        
        # Generate image
        result_image = await model_manager.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Image generated successfully in {processing_time:.2f}s")
        
        return ImageResponse(
            success=True,
            image=result_image,
            message="Image generated successfully",
            metadata={
                "prompt": request.prompt,
                "dimensions": f"{request.width}x{request.height}",
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Image generation failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        return ImageResponse(
            success=False,
            image=None,
            message=error_msg,
            processing_time=processing_time
        )


@router.post("/edit", response_model=ImageResponse)
async def edit_image(
    request: EditImageRequest,
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Edit image using text prompt with Qwen-Image editing model."""
    start_time = time.time()
    
    try:
        logger.info(f"Editing image with prompt: {request.prompt[:100]}...")
        
        # Edit image
        result_image = await model_manager.edit_image(
            image_base64=request.image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Image edited successfully in {processing_time:.2f}s")
        
        return ImageResponse(
            success=True,
            image=result_image,
            message="Image edited successfully",
            metadata={
                "prompt": request.prompt,
                "steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "strength": request.strength,
                "seed": request.seed
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Image editing failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        return ImageResponse(
            success=False,
            image=None,
            message=error_msg,
            processing_time=processing_time
        )


@router.get("/status")
async def get_status(model_manager: QwenImageManager = Depends(get_model_manager)):
    """Get detailed server status information."""
    try:
        status = await model_manager.get_detailed_status()
        return {
            "server_status": "running",
            "timestamp": time.time(),
            **status
        }
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get server status: {str(e)}"
        )


# Utility functions for multipart uploads
async def process_uploaded_image(file: UploadFile) -> str:
    """Convert uploaded file to base64 string."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only images are allowed."
        )
    
    # Check file size (max 10MB)
    file_size = 0
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum allowed: 10MB"
        )
    
    # Validate image format
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify it's a valid image
        
        # Re-open for processing (verify() consumes the file)
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95, optimize=True)
        buffer.seek(0)
        
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return base64_string
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


# Multipart upload endpoints
@router.post("/upload/edit", response_model=FileUploadResponse)
async def upload_edit_image(
    file: UploadFile = File(..., description="Image file to edit"),
    prompt: str = Form(..., description="Edit instruction prompt"),
    negative_prompt: str = Form("", description="Negative prompt"),
    num_inference_steps: int = Form(50, description="Number of denoising steps"),
    guidance_scale: float = Form(4.0, description="Guidance scale"),
    strength: float = Form(0.8, description="Editing strength"),
    seed: int = Form(None, description="Random seed"),
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Edit uploaded image using AI with multipart form data."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing uploaded file: {file.filename} ({file.content_type})")
        
        # Process uploaded image
        image_base64 = await process_uploaded_image(file)
        
        # Validate form parameters
        request_data = MultipartEditRequest(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed if seed is not None else None
        )
        
        logger.info(f"Editing image with prompt: {prompt[:100]}...")
        
        # Edit image using the model manager
        result_image = await model_manager.edit_image(
            image_base64=image_base64,
            prompt=request_data.prompt,
            negative_prompt=request_data.negative_prompt,
            num_inference_steps=request_data.num_inference_steps,
            guidance_scale=request_data.guidance_scale,
            strength=request_data.strength,
            seed=request_data.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Image edited successfully in {processing_time:.2f}s")
        
        return FileUploadResponse(
            success=True,
            image=result_image,
            message="Image edited successfully",
            metadata={
                "prompt": request_data.prompt,
                "steps": request_data.num_inference_steps,
                "guidance_scale": request_data.guidance_scale,
                "strength": request_data.strength,
                "seed": request_data.seed
            },
            processing_time=processing_time,
            file_info={
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(await file.read()) if hasattr(file, 'read') else 0
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Image editing failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@router.post("/upload/background-replacement", response_model=FileUploadResponse)
async def upload_background_replacement(
    file: UploadFile = File(..., description="Image file for background replacement"),
    prompt: str = Form(..., description="Background description prompt"),
    negative_prompt: str = Form("", description="Negative prompt"),
    num_inference_steps: int = Form(30, description="Number of denoising steps"),
    guidance_scale: float = Form(7.5, description="Guidance scale"),
    strength: float = Form(0.8, description="Background replacement strength"),
    seed: int = Form(None, description="Random seed"),
    model_manager: QwenImageManager = Depends(get_model_manager)
):
    """Replace background of uploaded image using AI with multipart form data."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing uploaded file for background replacement: {file.filename}")
        
        # Process uploaded image
        image_base64 = await process_uploaded_image(file)
        
        # Validate form parameters
        request_data = MultipartBackgroundRequest(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed if seed is not None else None
        )
        
        logger.info(f"Replacing background with: {prompt[:100]}...")
        
        # Use edit_image method for background replacement
        # In production, this might use a specialized background replacement model
        result_image = await model_manager.edit_image(
            image_base64=image_base64,
            prompt=f"Replace background with: {request_data.prompt}",
            negative_prompt=request_data.negative_prompt,
            num_inference_steps=request_data.num_inference_steps,
            guidance_scale=request_data.guidance_scale,
            strength=request_data.strength,
            seed=request_data.seed
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Background replacement completed in {processing_time:.2f}s")
        
        return FileUploadResponse(
            success=True,
            image=result_image,
            message="Background replacement completed successfully",
            metadata={
                "prompt": request_data.prompt,
                "task_type": "background_replacement",
                "steps": request_data.num_inference_steps,
                "guidance_scale": request_data.guidance_scale,
                "strength": request_data.strength,
                "seed": request_data.seed
            },
            processing_time=processing_time,
            file_info={
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(image_base64) * 3 // 4  # Estimate original size
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Background replacement failed: {str(e)}"
        logger.error(f"{error_msg} (after {processing_time:.2f}s)")
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )