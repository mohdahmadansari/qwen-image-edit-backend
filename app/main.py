"""
Qwen Image Edit Backend - FastAPI Main Module
API endpoints for image generation
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv
from PIL import Image
import io

from .schemas import GenerateRequest, GenerateResponse, HealthResponse, ImageFormat
from .inference import get_pipeline, QwenImagePipeline
from .utils import decode_base64_image, encode_image_base64, get_gpu_info, resize_image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    # Startup: Load the model
    logger.info("Starting up - Loading Qwen Image Edit pipeline...")
    try:
        pipeline = get_pipeline()
        pipeline.load()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        # Continue anyway - health check will report model not loaded
    
    yield
    
    # Shutdown: Unload the model
    logger.info("Shutting down - Unloading pipeline...")
    pipeline = get_pipeline()
    pipeline.unload()


# Create FastAPI app
app = FastAPI(
    title="Qwen Image Edit API",
    description="Backend API for Qwen Image Edit 2511 with Multiple-Angles LoRA",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Qwen Image Edit API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns model status and GPU information.
    """
    pipeline = get_pipeline()
    gpu_info = get_gpu_info()
    
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.is_loaded,
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info["gpu_name"],
        vram_free_gb=gpu_info["vram_free_gb"]
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    Generate an edited image from a reference image and prompt.
    
    The prompt should be finalized from the frontend - this endpoint
    does NOT modify or generate prompts.
    
    Request body:
    - prompt: Finalized diffusion-ready prompt string
    - reference_image: Base64 encoded reference image
    - seed: Optional random seed
    - guidance_scale: CFG scale (1.0-10.0)
    - num_inference_steps: Denoising steps (1-20)
    - height: Output height (256-768)
    - width: Output width (256-768)
    - output_format: png, jpeg, or base64
    """
    pipeline = get_pipeline()
    
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization."
        )
    
    if not request.reference_image:
        raise HTTPException(
            status_code=400,
            detail="reference_image is required"
        )
    
    try:
        # Decode reference image
        ref_image = decode_base64_image(request.reference_image)
        logger.info(f"Reference image decoded: {ref_image.size}")
        
        # Generate image
        output_image, seed_used = pipeline.generate(
            prompt=request.prompt,
            reference_image=ref_image,
            seed=request.seed,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            height=request.height,
            width=request.width
        )
        
        # Encode output
        if request.output_format == ImageFormat.BASE64:
            image_data = encode_image_base64(output_image, "PNG")
        elif request.output_format == ImageFormat.PNG:
            image_data = encode_image_base64(output_image, "PNG")
        else:
            image_data = encode_image_base64(output_image, "JPEG")
        
        logger.info(f"Generation successful: seed={seed_used}")
        
        return GenerateResponse(
            success=True,
            image=image_data,
            seed=seed_used
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerateResponse(
            success=False,
            error=str(e)
        )


@app.post("/generate-upload")
async def generate_from_upload(
    prompt: Annotated[str, Form(description="Finalized prompt string")],
    reference_image: Annotated[UploadFile, File(description="Reference image file")],
    seed: Annotated[int | None, Form()] = None,
    guidance_scale: Annotated[float, Form()] = 1.0,
    num_inference_steps: Annotated[int, Form()] = 4,
    height: Annotated[int, Form()] = 768,
    width: Annotated[int, Form()] = 768
):
    """
    Generate an edited image from an uploaded file.
    Alternative to /generate for form-based uploads.
    """
    pipeline = get_pipeline()
    
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization."
        )
    
    try:
        # Read and decode uploaded image
        image_data = await reference_image.read()
        ref_image = Image.open(io.BytesIO(image_data))
        
        if ref_image.mode != "RGB":
            ref_image = ref_image.convert("RGB")
        
        logger.info(f"Uploaded image: {ref_image.size}")
        
        # Generate image
        output_image, seed_used = pipeline.generate(
            prompt=prompt,
            reference_image=ref_image,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
        )
        
        # Return as PNG bytes
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"X-Seed-Used": str(seed_used)}
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true"
    )
