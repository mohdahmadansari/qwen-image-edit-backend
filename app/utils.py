"""
Qwen Image Edit Backend - Utility Functions
Image processing and helper utilities
"""

import base64
import io
from PIL import Image
from typing import Optional, Tuple
import torch


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Output format (PNG or JPEG)
        
    Returns:
        Base64 encoded string with data URL prefix
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    
    return f"data:{mime_type};base64,{base64_data}"


def resize_image(
    image: Image.Image,
    max_size: int = 768,
    multiple_of: int = 64
) -> Tuple[Image.Image, int, int]:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    Dimensions are adjusted to be multiples of 64 for diffusion models.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height)
        multiple_of: Ensure dimensions are multiples of this value
        
    Returns:
        Tuple of (resized_image, new_width, new_height)
    """
    width, height = image.size
    
    # Calculate scale to fit within max_size
    scale = min(max_size / width, max_size / height)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        new_width = width
        new_height = height
    
    # Round to multiple of 64
    new_width = (new_width // multiple_of) * multiple_of
    new_height = (new_height // multiple_of) * multiple_of
    
    # Ensure minimum size
    new_width = max(new_width, multiple_of)
    new_height = max(new_height, multiple_of)
    
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image, new_width, new_height


def get_gpu_info() -> dict:
    """
    Get GPU information for health checks.
    
    Returns:
        Dictionary with GPU availability, name, and free VRAM
    """
    info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_free_gb": None
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # Get free VRAM
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total - allocated
        info["vram_free_gb"] = round(free / (1024**3), 2)
    
    return info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
