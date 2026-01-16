"""
Qwen Image Edit Backend - Inference Pipeline
Handles model loading and image generation
"""

import os
import logging
from typing import Optional
import torch
from PIL import Image
from diffusers import DiffusionPipeline

from .utils import resize_image, clear_gpu_memory

logger = logging.getLogger(__name__)


class QwenImagePipeline:
    """
    Inference pipeline for Qwen Image Edit 2511 with Multiple-Angles LoRA.
    Optimized for RTX 4070 Laptop (8GB VRAM).
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit-2511",
        lora_id: str = "multimodalart/qwen-image-multiple-angles",
        lora_scale: float = 1.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_id: HuggingFace model ID for base model
            lora_id: HuggingFace model ID for LoRA adapter
            lora_scale: LoRA adapter scale (0.0-1.0)
            device: Device to use (cuda or cpu)
            dtype: Data type (float16 for VRAM optimization)
        """
        self.model_id = model_id
        self.lora_id = lora_id
        self.lora_scale = lora_scale
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self._is_loaded = False
    
    def load(self) -> None:
        """Load the model and LoRA adapter with memory optimizations."""
        if self._is_loaded:
            logger.info("Pipeline already loaded")
            return
        
        logger.info(f"Loading base model: {self.model_id}")
        
        # Load base pipeline - use DiffusionPipeline to auto-detect correct class
        # Use bfloat16 for numerical stability (per official example)
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,  # bfloat16 required for Qwen
        )
        
        # Load LoRA adapter if specified
        if self.lora_id:
            logger.info(f"Loading LoRA adapter: {self.lora_id}")
            try:
                self.pipe.load_lora_weights(self.lora_id)
            except Exception as e:
                logger.warning(f"Could not load LoRA: {e}")
        
        # Enable CPU offload for 8GB VRAM (instead of .to(device))
        # This keeps model on CPU and moves layers to GPU only when needed
        logger.info("Enabling sequential CPU offload for 8GB VRAM...")
        self.pipe.enable_sequential_cpu_offload()
        
        # Enable memory optimizations
        self._enable_memory_optimizations()
        
        self._is_loaded = True
        logger.info("Pipeline loaded successfully with CPU offload")
    
    def _enable_memory_optimizations(self) -> None:
        """Enable memory optimizations for limited VRAM."""
        if self.pipe is None:
            return
        
        # Enable attention slicing for reduced memory
        self.pipe.enable_attention_slicing(slice_size="auto")
        
        # Enable VAE slicing for large images
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        # Enable VAE tiling for very large images
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()
        
        # Enable model CPU offload if still running low on memory
        # self.pipe.enable_model_cpu_offload()
        
        logger.info("Memory optimizations enabled: attention_slicing, vae_slicing")
    
    def generate(
        self,
        prompt: str,
        reference_image: Image.Image,
        seed: Optional[int] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        height: int = 768,
        width: int = 768
    ) -> tuple[Image.Image, int]:
        """
        Generate an edited image.
        
        Args:
            prompt: Finalized diffusion-ready prompt
            reference_image: Reference image for editing
            seed: Random seed (None for random)
            guidance_scale: CFG scale
            num_inference_steps: Number of denoising steps
            height: Output height (max 768)
            width: Output width (max 768)
            
        Returns:
            Tuple of (generated_image, seed_used)
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        
        # Resize reference image to match output dimensions
        ref_resized, _, _ = resize_image(
            reference_image,
            max_size=max(height, width),
            multiple_of=64
        )
        
        # Setup generator with seed
        # Use CPU generator since we're using sequential CPU offload
        if seed is None:
            seed = torch.randint(0, 2147483647, (1,)).item()
        
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logger.info(f"Generating with seed={seed}, steps={num_inference_steps}, cfg={guidance_scale}")
        
        # Clear memory before generation
        clear_gpu_memory()
        
        # Run inference with parameters matching official HuggingFace example
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                image=[ref_resized],  # Must be a list!
                height=height,
                width=width,
                guidance_scale=1.0,  # Fixed at 1.0 per docs
                true_cfg_scale=4.0,  # Required for CFG
                negative_prompt=" ",  # Required even if empty
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generator,
            )
        
        output_image = result.images[0]
        
        # Clear memory after generation
        clear_gpu_memory()
        
        return output_image, seed
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            clear_gpu_memory()
            self._is_loaded = False
            logger.info("Pipeline unloaded")
    
    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._is_loaded


# Global pipeline instance
_pipeline: Optional[QwenImagePipeline] = None


def get_pipeline() -> QwenImagePipeline:
    """Get the global pipeline instance (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QwenImagePipeline(
            model_id=os.getenv("MODEL_ID", "Qwen/Qwen-Image-Edit-2511"),
            lora_id=os.getenv("LORA_ID", "multimodalart/qwen-image-multiple-angles"),
            lora_scale=float(os.getenv("LORA_SCALE", "1.0")),
            device=os.getenv("DEVICE", "cuda"),
            dtype=torch.float16 if os.getenv("DTYPE", "float16") == "float16" else torch.float32
        )
    return _pipeline
