"""
Qwen Image Edit Backend - Pydantic Schemas
Request and response models for the API
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ImageFormat(str, Enum):
    """Output image format"""
    PNG = "png"
    JPEG = "jpeg"
    BASE64 = "base64"


class GenerateRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(
        ...,
        description="Finalized diffusion-ready prompt string from frontend",
        example="<sks> right side view high angle shot close-up shot, studio lighting, realistic perspective, same subject, consistent identity"
    )
    reference_image: Optional[str] = Field(
        None,
        description="Base64 encoded reference image (alternative to file upload)"
    )
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2147483647,
        description="Random seed for reproducibility. If None, random seed is used."
    )
    guidance_scale: float = Field(
        1.0,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance scale"
    )
    num_inference_steps: int = Field(
        4,
        ge=1,
        le=20,
        description="Number of denoising steps"
    )
    height: int = Field(
        768,
        ge=256,
        le=768,
        description="Output image height (max 768 for VRAM optimization)"
    )
    width: int = Field(
        768,
        ge=256,
        le=768,
        description="Output image width (max 768 for VRAM optimization)"
    )
    output_format: ImageFormat = Field(
        ImageFormat.BASE64,
        description="Output format: png, jpeg, or base64"
    )


class GenerateResponse(BaseModel):
    """Response model for image generation"""
    success: bool = Field(..., description="Whether generation succeeded")
    image: Optional[str] = Field(None, description="Generated image (base64 or URL)")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    model_loaded: bool = False
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    vram_free_gb: Optional[float] = None
