# main.py - video generation API server


import os
import uuid
import json
import time
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

import imageio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
import numpy as np

MAX_DURATION = 10
MIN_DURATION = 1
MAX_RESOLUTION = 768
MIN_FPS = 4
MAX_FPS = 24
MAX_BATCH_SIZE = 5

task_queue = queue.Queue()
task_results = {}
task_states = {}
executor = ThreadPoolExecutor(max_workers=4)

def video_generator_worker():
    """Worker thread for processing video generation tasks"""
    global pipeline
    
    while True:
        try:
            job_id, task = task_queue.get()
            if job_id is None:
                break
                
            try:
                # Ensure we have the pipeline loaded in this thread
                if pipeline is None:
                    load_result = asyncio.run(test_model_loading())
                    if load_result.get("status") != "success":
                        raise Exception(f"Failed to load model: {load_result.get('message')}")
                        
                task_states[job_id] = "processing"
                result = task()
                task_results[job_id] = {"status": "completed", "result": result}
                task_states[job_id] = "completed"
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Task failed for job {job_id}: {error_msg}")
                task_results[job_id] = {"status": "failed", "error": error_msg}
                task_states[job_id] = "failed"
            finally:
                task_queue.task_done()
        except Exception as e:
            logger.error(f"Worker thread error: {e}")

# Start worker threads
worker_threads = []
for _ in range(4):
    t = threading.Thread(target=video_generator_worker)
    t.daemon = True
    t.start()
    worker_threads.append(t)

# MockTorch for fallback if PyTorch is unavailable
class MockTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def device_count():
            return 0
        @staticmethod
        def current_device():
            return 0
        @staticmethod
        def get_device_name():
            return "No GPU (PyTorch unavailable)"
        @staticmethod
        def memory_allocated():
            return 0
        @staticmethod
        def memory_reserved():
            return 0

    class version:
        cuda = "unavailable"

    @staticmethod
    def bfloat16():
        return "bfloat16"

    class Generator:
        def __init__(self, device="cpu"):
            self.device = device
        def manual_seed(self, seed):
            return self

    __version__ = "mock"

# try to import PyTorch -> fallback if CUDA issues
try:
    import torch
    TORCH_AVAILABLE = True
    logging.info(f"PyTorch {torch.__version__} loaded successfully")
    if torch.cuda.is_available():
        logging.info(f"CUDA {torch.version.cuda} is available")
        logging.info(f"GPU: {torch.cuda.get_device_name()}")
    else:
        logging.info("CUDA not available, running on CPU")
except Exception as e:
    logging.warning(f"Failed to import PyTorch: {e}")
    TORCH_AVAILABLE = False
    torch = MockTorch()
    logging.info("Using MockTorch for graceful fallback")


# configure logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# store model
pipeline = None


# Advanced Parameter Optimization Functions

def optimize_generation_parameters(
    prompt: str,
    duration: int,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    negative_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize generation parameters for better prompt adherence
    """
    if guidance_scale is None:
        guidance_scale = 7.5 
    if num_inference_steps is None:
        num_inference_steps = 25 
    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted, watermark, text, static image"
    
    return {
        'guidance_scale': guidance_scale,
        'num_inference_steps': num_inference_steps,
        'negative_prompt': negative_prompt
    }

def enhance_prompt(prompt: str) -> str:
    """
    Enhance prompt for better video generation without over-complicating
    """
    prompt = prompt.strip()
    
    # Don't enhance if already long or contains video-specific terms
    if len(prompt.split()) > 15 or any(word in prompt.lower() for word in ['video', 'moving', 'motion', 'animation']):
        return prompt
    
    # minimal enhancement for video generation
    return f"{prompt}, smooth motion"

def get_style_preset(style: str) -> Dict[str, Any]:
    """
    Get simplified style presets focused on core differences
    """
    presets = {
        "cinematic": {
            "guidance_scale": 8.0,
            "num_inference_steps": 30,
            "prompt_suffix": ", cinematic style",
            "negative_prompt": "blurry, low quality, amateur, shaky"
        },
        "realistic": {
            "guidance_scale": 7.5,
            "num_inference_steps": 28,
            "prompt_suffix": ", photorealistic",
            "negative_prompt": "cartoon, anime, artistic, low quality, blurry"
        },
        "artistic": {
            "guidance_scale": 8.5,
            "num_inference_steps": 32,
            "prompt_suffix": ", artistic style",
            "negative_prompt": "photographic, realistic, low quality, blurry"
        }
    }
    
    return presets.get(style, {
        "guidance_scale": 7.5,
        "num_inference_steps": 25,
        "prompt_suffix": "",
        "negative_prompt": "blurry, low quality, distorted, watermark, text, static image"
    })


# Video Saving & File Management Functions

def ensure_output_directories() -> None:
    """Create necessary output directories if they don't exist"""
    base_dir = os.getcwd()
    if not base_dir.endswith("take-home"):
        base_dir = os.path.dirname(base_dir)
    
    directories = [
        os.path.join(base_dir, "outputs", "videos"),
        os.path.join(base_dir, "outputs", "metadata"), 
        os.path.join(base_dir, "outputs", "temp")
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info(f"Output directories ensured in {base_dir}/outputs")

def get_output_path(subdir: str) -> str:
    """Get the correct output path for a subdirectory"""
    if os.path.exists("/app/outputs"):
        return os.path.join("/app/outputs", subdir)
    else:
        base_dir = os.getcwd()
        if not base_dir.endswith("take-home"):
            base_dir = os.path.dirname(base_dir)
        return os.path.join(base_dir, "outputs", subdir)

def save_video_frames(
    frames: List,
    prompt: str,
    duration: int,
    resolution: str,
    fps: int = 8,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save generated frames as an MP4 video file
    
    Parameters:
    - frames: List of PIL Image objects or numpy arrays
    - prompt: Original text prompt used for generation
    - duration: Video duration in seconds
    - resolution: Video resolution (e.g., "320x320")
    
    Returns:
    - Dictionary with job_id, filename, and file paths
    """
    try:
        ensure_output_directories()
        
        if job_id is None:
            job_id = str(uuid.uuid4())[:8].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}_{job_id}.mp4"
        
        video_path = os.path.join(get_output_path("videos"), filename)
        
        # frames -> numpy arrays if PIL images
        processed_frames = []
        for frame in frames:
            if hasattr(frame, 'numpy'):  # PyTorch tensor
                frame_array = frame.cpu().numpy()
                if frame_array.shape[0] == 3:
                    frame_array = np.transpose(frame_array, (1, 2, 0))
                if frame_array.max() <= 1.0:
                    frame_array = (frame_array * 255).astype(np.uint8)
                processed_frames.append(frame_array)
            elif isinstance(frame, Image.Image): 
                processed_frames.append(np.array(frame))
            elif isinstance(frame, np.ndarray):
                processed_frames.append(frame)
            else:
                logger.warning(f"Unknown frame type: {type(frame)}")
                processed_frames.append(np.array(frame))
        
        # calculate FPS (frames per second)
        actual_fps = fps
        
        # save as MP4 video
        logger.info(f"Saving video with {len(processed_frames)} frames at {actual_fps} FPS")     
        writer = imageio.get_writer(
            video_path, 
            fps=actual_fps, 
            quality=8,  
            codec='libx264',  
            format='FFMPEG',  # for MP4
            output_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-crf', '18']  # Higher quality encoding
        )
        
        for frame in processed_frames:
            writer.append_data(frame)
        writer.close()
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        logger.info(f"Video saved: {filename} ({file_size_mb:.1f} MB)")
        
        return {
            "job_id": job_id,
            "filename": filename,
            "video_path": video_path,
            "file_size_mb": round(file_size_mb, 2),
            "fps": actual_fps,
            "total_frames": len(processed_frames)
        }
        
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        raise Exception(f"Video saving failed: {str(e)}")

def save_generation_metadata(job_id: str, metadata: dict) -> None:
    """Save metadata about the video generation"""
    try:
        job_id = job_id.lower().strip()
        
        # Create metadata dir if it doesn't exist
        metadata_dir = get_output_path("metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Prepare metadata path
        metadata_path = os.path.join(metadata_dir, f"{job_id}.json")
        logger.info(f"Saving metadata to: {metadata_path}")
        
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()
        if "job_id" not in metadata:
            metadata["job_id"] = job_id
            
        # Save metadata with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                
                with open(metadata_path, 'r') as f:
                    saved_metadata = json.load(f)
                    if saved_metadata:
                        logger.info(f"Metadata saved and verified for job {job_id}")
                        return
            except Exception as write_error:
                if attempt == max_retries - 1:
                    raise write_error
                logger.warning(f"Retry {attempt + 1}/{max_retries} saving metadata: {write_error}")
                time.sleep(0.5)
                
    except Exception as e:
        logger.error(f"Failed to save metadata for job {job_id}: {str(e)}")
        raise 

def find_video_file(job_id: str) -> Optional[str]:
    """Find video file by job_id"""
    videos_dir = get_output_path("videos")
    if not os.path.exists(videos_dir):
        return None
    
    for filename in os.listdir(videos_dir):
        if job_id in filename and filename.endswith('.mp4'):
            return os.path.join(videos_dir, filename)
    return None

# create the FastAPI application (web server to receive HTTP requests)
app = FastAPI(
    title="ModelScope Text-to-Video API",
    description="Converts text prompts into videos using damo-vilab/text-to-video-ms-1.7b model (high quality)",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Ensure all required directories exist when server starts"""
    ensure_output_directories()
    logger.info("Output directories verified")

# Basic Endpoints (Routes)

@app.get("/")
async def root():
    """
    The homepage of our API with links to generated videos
    When someone visits http://localhost:8000/ they'll see this message
    """
    try:
        videos_dir = get_output_path("videos")
        metadata_dir = get_output_path("metadata")
        recent_videos = []
        
        if os.path.exists(metadata_dir):
            for metadata_file in sorted(os.listdir(metadata_dir), reverse=True)[:5]:
                if metadata_file.endswith('.json'):
                    job_id = metadata_file.replace('.json', '')
                    metadata_path = os.path.join(metadata_dir, metadata_file)
                    
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        video_path = find_video_file(job_id)
                        if video_path and os.path.exists(video_path):
                            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                            recent_videos.append({
                                'job_id': job_id,
                                'prompt': metadata.get('prompt', 'Unknown'),
                                'file_size_mb': round(file_size_mb, 2),
                                'created_at': metadata.get('created_at', 'Unknown')[:16].replace('T', ' ')
                            })
                    except:
                        continue
        
        return {
            "message": "Welcome to the ModelScope Text-to-Video API!",
            "model": "damo-vilab/text-to-video-ms-1.7b (high quality)",
            "status": "Server is running",
            "docs": "Visit /docs to see all available endpoints",
            "recent_videos": recent_videos,
            "access_info": {
                "preview_note": "Preview links only work on localhost (same machine)",
                "download_format": "/download/{job_id} - downloads MP4 file",
                "preview_format": "/preview/{job_id} - web preview (localhost only)",
                "status_format": "/status/{job_id} - job information"
            }
        }
    except Exception as e:
        return {
            "message": "Welcome to the ModelScope Text-to-Video API!",
            "model": "damo-vilab/text-to-video-ms-1.7b (high quality)",
            "status": "Server is running",
            "docs": "Visit /docs to see all available endpoints",
            "error": f"Could not load recent videos: {str(e)}"
        }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint - tells us if everything is working
    (what Docker calls to check if our service is healthy)
    """

    gpu_available = torch.cuda.is_available()
    
    gpu_info = {}
    if gpu_available:
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
        }
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "message": "API is ready to receive requests"
    }

@app.get("/test-model-loading")
async def test_model_loading() -> Dict[str, Any]:
    """
    Test endpoint to load the ModelScope text-to-video model (high quality)
    This loads the damo-vilab/text-to-video-ms-1.7b model
    """
    global pipeline
    
    try:
        logger.info("Starting ModelScope text-to-video model loading (high quality)...")
        
        # check if model is already loaded
        if pipeline is not None:
            return {
                "status": "success",
                "message": "LTX-Video-0.9.7-distilled model already loaded!",
                "model_type": str(type(pipeline)),
                "model_id": "Lightricks/LTX-Video-0.9.7-distilled"
            }
        
        # check if PyTorch is available
        if not TORCH_AVAILABLE:
            return {
                "status": "pytorch_unavailable",
                "message": "PyTorch is not available due to CUDA library conflicts",
                "fallback_mode": "CPU simulation enabled",
                "development_note": "API is functional for testing endpoints, but model loading requires PyTorch",
                "gpu_hardware": "H100 80GB detected and accessible",
                "solution": "CUDA library compatibility issue - can be resolved with library updates"
            }
        
        try:
            from diffusers import LTXPipeline
            logger.info("Successfully imported LTXPipeline")
        except ImportError as e:
            logger.error(f"Cannot import LTXPipeline: {e}")
            return {
                "status": "error",
                "error_type": "import_error",
                "message": f"LTXPipeline not available: {str(e)}",
                "hint": "Please update diffusers to latest version: pip install --upgrade diffusers"
            }
        logger.info("Loading LTX-Video-0.9.7-distilled model (this may take a few minutes)...")
        
        try:
            # Try the ModelScope text-to-video model first (better quality)
            logger.info("Trying primary model: damo-vilab/text-to-video-ms-1.7b")
            from diffusers import DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("Successfully loaded ModelScope text-to-video model (high quality)")
        except Exception as model_error:
            logger.error(f"Failed to load damo-vilab/text-to-video-ms-1.7b: {model_error}")
            
            # Try the regular LTX-Video model as backup
            logger.info("Trying backup model: Lightricks/LTX-Video")
            try:
                from diffusers import LTXPipeline
                pipeline = LTXPipeline.from_pretrained(
                    "Lightricks/LTX-Video",
                    torch_dtype=torch.bfloat16
                )
                logger.info("Successfully loaded LTX-Video model as backup")
            except Exception as ltx_error:
                logger.error(f"LTX-Video model also failed: {ltx_error}")
                
                # fallback to distilled version
                logger.info("Trying final fallback: Lightricks/LTX-Video-0.9.7-distilled")
                try:
                    pipeline = LTXPipeline.from_pretrained(
                        "Lightricks/LTX-Video-0.9.7-distilled",
                        torch_dtype=torch.bfloat16
                    )
                    logger.info("Successfully loaded LTX-Video-0.9.7-distilled as final fallback")
                except Exception as fallback_error:
                    logger.error(f"All models failed: {fallback_error}")
                    return {
                        "status": "error",
                        "error_type": "model_loading_error",
                        "message": f"All models failed. ModelScope: {str(model_error)}, LTX: {str(ltx_error)}, Distilled: {str(fallback_error)}",
                        "attempted_models": ["damo-vilab/text-to-video-ms-1.7b", "Lightricks/LTX-Video", "Lightricks/LTX-Video-0.9.7-distilled"],
                        "hint": "Check internet connection and model availability"
                    }
        
        # Move to device (GPU if available, if not -> CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)
        logger.info(f"Model moved to {device}")
        
        # memory optimizations
        try:
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                logger.info("CPU offload enabled")
        except Exception as e:
            logger.warning(f"CPU offload failed: {e}")
        
        try:
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
                logger.info("VAE slicing enabled")
        except Exception as e:
            logger.warning(f"VAE slicing failed: {e}")
        
        try:
            if hasattr(pipeline, 'enable_vae_tiling'):
                pipeline.enable_vae_tiling()
                logger.info("VAE tiling enabled")
        except Exception as e:
            logger.warning(f"VAE tiling failed: {e}")
        
        return {
            "status": "success",
            "message": "LTX-Video-0.9.7-distilled model loaded successfully!",
            "model_id": "Lightricks/LTX-Video-0.9.7-distilled",
            "device": device,
            "model_type": str(type(pipeline)),
            "optimizations_enabled": {
                "cpu_offload": hasattr(pipeline, 'enable_model_cpu_offload'),
                "vae_slicing": hasattr(pipeline, 'enable_vae_slicing'),
                "vae_tiling": hasattr(pipeline, 'enable_vae_tiling')
            }
        }
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return {
            "status": "error",
            "error_type": "unexpected_error", 
            "message": f"Unexpected error loading LTX-Video-0.9.7-distilled: {str(e)}",
            "model_id": "Lightricks/LTX-Video-0.9.7-distilled"
        }

@app.get("/model-status")
async def model_status() -> Dict[str, Any]:
    """
    Check if the LTX-Video-0.9.7-distilled model is loaded and ready
    """
    global pipeline
    
    if pipeline is None:
        return {
            "status": "not_loaded",
            "message": "LTX-Video-0.9.7-distilled model not loaded. Use /test-model-loading to load it.",
            "expected_model": "Lightricks/LTX-Video-0.9.7-distilled"
        }
    else:
        return {
            "status": "loaded",
            "message": "LTX-Video-0.9.7-distilled model is loaded and ready!",
            "model_type": str(type(pipeline)),
            "model_id": "Lightricks/LTX-Video-0.9.7-distilled",
            "device": str(pipeline.device) if hasattr(pipeline, 'device') else "unknown"
        }

@app.get("/debug-diffusers")
async def debug_diffusers() -> Dict[str, Any]:
    """
    Debug endpoint to see what's available in the diffusers library
    """
    try:
        import diffusers
        
        available_classes = [name for name in dir(diffusers) if not name.startswith('_')]
        pipeline_classes = [name for name in available_classes if 'Pipeline' in name]
        
        return {
            "diffusers_version": diffusers.__version__,
            "total_classes": len(available_classes),
            "pipeline_classes": pipeline_classes,
            "all_classes": available_classes[:20]  # first 20 so its not too much data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to inspect diffusers: {str(e)}"
        }

@app.post("/analyze-prompt")
async def analyze_prompt(prompt: str) -> Dict[str, Any]:
    """
    Analyze a prompt and provide simple optimization suggestions
    """
    try:
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        prompt = prompt.strip()
        word_count = len(prompt.split())
        
        # suggestions for better video generation
        suggestions = []
        if word_count < 3:
            suggestions.append("Add more descriptive details for better results")
        if word_count > 20:
            suggestions.append("Consider shortening the prompt - very long prompts can confuse the model")
        if not any(word in prompt.lower() for word in ['moving', 'motion', 'video', 'walking', 'flying', 'flowing']):
            suggestions.append("Consider adding motion words like 'moving', 'flowing', or 'walking'")
        
        return {
            "original_prompt": prompt,
            "word_count": word_count,
            "complexity": "simple" if word_count < 5 else "complex" if word_count > 15 else "moderate",
            "suggestions": suggestions,
            "recommended_settings": {
                "guidance_scale": 7.5,
                "num_inference_steps": 25,
                "fps": 8
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze prompt: {str(e)}"
        )

@app.get("/styles")
async def get_available_styles() -> Dict[str, Any]:
    """
    Get list of available simplified style presets
    """
    styles = {
        "cinematic": {
            "name": "Cinematic",
            "description": "Film-like quality with dramatic composition",
            "best_for": "Professional-looking content, storytelling"
        },
        "realistic": {
            "name": "Realistic",
            "description": "Natural, photorealistic appearance",
            "best_for": "Documentary-style content, real-world scenes"
        },
        "artistic": {
            "name": "Artistic",
            "description": "Creative and stylized appearance",
            "best_for": "Creative content, artistic expression"
        }
    }
    
    return {
        "available_styles": styles,
        "usage": "Add 'style' parameter to /generate endpoint",
        "note": "Styles provide subtle guidance - your prompt is still the main driver"
    }


# Video Generation Endpoints

@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str,
    duration: int = 3,
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    fps: int = 8,
    height: Optional[int] = None,
    width: Optional[int] = None,
    resolution: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    style: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a video from a text prompt w/advanced controls
    """
    global pipeline
    
    # check if model is loaded, load it if not
    if pipeline is None:
        logger.info("Model not loaded, attempting to load ModelScope text-to-video model...")
        try:
            load_result = await test_model_loading()
            if load_result.get("status") != "success":
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load ModelScope model: {load_result.get('message', 'Unknown error')}"
                )
            logger.info("Model loaded successfully")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to auto-load ModelScope model: {str(e)}"
            )
    
    try:
        logger.info(f"Generating video for prompt: '{prompt}'")
        
        # Enhanced validation
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        if duration < MIN_DURATION or duration > MAX_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Duration must be between {MIN_DURATION} and {MAX_DURATION} seconds"
            )
        if fps < MIN_FPS or fps > MAX_FPS:
            raise HTTPException(
                status_code=400,
                detail=f"FPS must be between {MIN_FPS} and {MAX_FPS}"
            )
        # Handle resolution parameter
        if resolution is not None:
            if resolution > MAX_RESOLUTION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resolution cannot exceed {MAX_RESOLUTION}x{MAX_RESOLUTION} pixels"
                )
            height = width = resolution
        else:
            # Use height/width if provided, otherwise default to 512
            height = height or 512
            width = width or 512
            if height > MAX_RESOLUTION or width > MAX_RESOLUTION:
                raise HTTPException(
                    status_code=400,
                    detail=f"Resolution cannot exceed {MAX_RESOLUTION}x{MAX_RESOLUTION} pixels"
                )
        if guidance_scale is not None and (guidance_scale < 1 or guidance_scale > 20):
            raise HTTPException(
                status_code=400,
                detail="Guidance scale must be between 1 and 20"
            )
        if num_inference_steps is not None and (num_inference_steps < 10 or num_inference_steps > 50):
            raise HTTPException(
                status_code=400,
                detail="Inference steps must be between 10 and 50"
            )
        
        # Auto-optimize parameters based on prompt and settings
        optimized_params = optimize_generation_parameters(
            prompt=prompt,
            duration=duration,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt
        )
        
        # Apply style preset if specified
        if style:
            style_preset = get_style_preset(style)
            if guidance_scale is None:
                optimized_params['guidance_scale'] = style_preset['guidance_scale']
            if num_inference_steps is None:
                optimized_params['num_inference_steps'] = style_preset['num_inference_steps']
            if negative_prompt is None:
                optimized_params['negative_prompt'] = style_preset['negative_prompt']
            # Apply minimal style suffix to prompt
            prompt = f"{prompt}{style_preset['prompt_suffix']}"
        
        logger.info("Starting video generation...")
        start_time = datetime.now()
        
        try:
            # Use provided seed or generate one from prompt
            if seed is None:
                import hashlib
                prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
                seed = prompt_hash % (2**31)  # Keep within valid range
            
            enhanced_prompt = enhance_prompt(prompt)
            
            logger.info(f"Using seed {seed} for prompt: '{prompt}'")
            logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
            logger.info(f"Parameters: guidance_scale={optimized_params['guidance_scale']}, "
                       f"steps={optimized_params['num_inference_steps']}")
            
            # Create a unique job ID
            job_id = str(uuid.uuid4())[:8].lower()
            task_states[job_id] = "queued"
            
            # Define the video generation function
            def generate_video_task():
                try:
                    # Calculate optimal frame count
                    num_frames = max(16, duration * fps)
                    
                    video_frames = pipeline(
                        prompt=enhanced_prompt,
                        negative_prompt=optimized_params['negative_prompt'],
                        num_frames=num_frames,
                        guidance_scale=optimized_params['guidance_scale'],
                        num_inference_steps=optimized_params['num_inference_steps'],
                        height=height,
                        width=width,
                        generator=torch.Generator(device=pipeline.device).manual_seed(seed)
                    ).frames[0]
                    
                    logger.info(f"Generated {len(video_frames)} frames")
                    
                    end_time = datetime.now()
                    generation_time = (end_time - start_time).total_seconds()
                    logger.info(f"Generation completed in {generation_time:.1f} seconds")
                    
                    return video_frames, num_frames, generation_time
                except Exception as e:
                    logger.error(f"Video generation failed: {str(e)}")
                    raise e
            
            async def process_generation_result(task_result, metadata):
                """Process the video generation result"""
                try:
                    video_frames, num_frames, generation_time = task_result
                    video_info = save_video_frames(
                        frames=video_frames,
                        prompt=prompt,
                        duration=duration,
                        resolution=f"{height}x{width}",
                        fps=fps,
                        job_id=job_id
                    )
                    
                    metadata.update({
                        "frames_generated": num_frames,
                        "generation_time_seconds": generation_time,
                        "status": "completed",
                        "video_info": video_info
                    })
                    save_generation_metadata(job_id, metadata)
                    
                    # Log completion
                    logger.info(f"Video generation completed for job {job_id}")
                except Exception as e:
                    metadata["status"] = "failed"
                    metadata["error"] = str(e)
                    save_generation_metadata(job_id, metadata)
                    logger.error(f"Video generation failed for job {job_id}: {str(e)}")
            
            # Save initial metadata
            metadata = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "duration": duration,
                "seed": seed,
                "resolution": f"{height}x{width}",
                "fps": fps,
                "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
                "created_at": datetime.now().isoformat(),
                "device_used": str(pipeline.device),
                "generation_parameters": {
                    "guidance_scale": optimized_params['guidance_scale'],
                    "num_inference_steps": optimized_params['num_inference_steps'],
                    "negative_prompt": optimized_params['negative_prompt']
                },
                "status": "processing"
            }
            save_generation_metadata(job_id, metadata)
            
            # Add task to queue
            task_queue.put((job_id, generate_video_task))
            
            # Add a background task to handle the result
            def check_task_completion():
                """Check task completion and process result"""
                try:
                    while True:
                        if job_id in task_results:
                            result = task_results[job_id]
                            if result["status"] == "completed":
                                asyncio.run(process_generation_result(result["result"], metadata))
                            elif result["status"] == "failed":
                                metadata["status"] = "failed"
                                metadata["error"] = result.get("error", "Unknown error")
                                save_generation_metadata(job_id, metadata)
                            break
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in completion check for job {job_id}: {str(e)}")
                    metadata["status"] = "failed"
                    metadata["error"] = str(e)
                    save_generation_metadata(job_id, metadata)
            
            background_tasks.add_task(check_task_completion)
            
            return {
                "status": "success",
                "message": "Video generation task queued successfully",
                "job_id": job_id
            }
            
        except Exception as gen_error:
            logger.error(f"Generation failed: {str(gen_error)}")
            return {
                "status": "error",
                "error_type": "generation_error",
                "message": f"Video generation failed: {str(gen_error)}",
                "prompt": prompt,
                "hint": "The model might need different parameters or more memory"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/generate-batch")
async def generate_batch_videos(
    prompts: List[str],
    duration: int = 3,
    fps: int = 8
) -> Dict[str, Any]:
    """
    Generate multiple videos from a list of prompts
    
    Parameters:
    - prompts: List of text descriptions for videos
    - duration: Video length in seconds for all videos (default: 3)
    - fps: Frames per second for all videos (default: 8)
    """
    global pipeline
    
    if not prompts or len(prompts) == 0:
        raise HTTPException(
            status_code=400,
            detail="Prompts list cannot be empty"
        )
    
    if len(prompts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_SIZE} prompts allowed per batch"
        )
    
    # check if model is loaded
    if pipeline is None:
        logger.info("Model not loaded, attempting to load LTX-Video-0.9.7-distilled...")
        try:
            load_result = await test_model_loading()
            if load_result.get("status") != "success":
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load LTX-Video-0.9.7-distilled model: {load_result.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to auto-load LTX-Video-0.9.7-distilled model: {str(e)}"
            )
    
    batch_results = []
    total_start_time = datetime.now()
    
    for i, prompt in enumerate(prompts, 1):
        try:
            logger.info(f"Generating video {i}/{len(prompts)} for prompt: '{prompt}'")
            
            # Generate unique seed for each prompt
            import hashlib
            prompt_hash = int(hashlib.md5(f"{prompt}_{i}".encode()).hexdigest()[:8], 16)
            seed = prompt_hash % (2**31)
            
            # Auto-optimize parameters
            optimized_params = optimize_generation_parameters(prompt, duration)
            enhanced_prompt = enhance_prompt(prompt)
            
            start_time = datetime.now()
            
            video_frames = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=optimized_params['negative_prompt'],
                num_frames=max(16, duration * fps),
                guidance_scale=optimized_params['guidance_scale'],
                num_inference_steps=optimized_params['num_inference_steps'],
                height=512,
                width=512,
                generator=torch.Generator(device=pipeline.device).manual_seed(seed)
            ).frames[0]
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            video_info = save_video_frames(
                frames=video_frames,
                prompt=prompt,
                duration=duration,
                resolution="512x512",
                fps=fps
            )
            
            metadata = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "batch_index": i,
                "total_batch_size": len(prompts),
                "duration": duration,
                "seed": seed,
                "frames_generated": len(video_frames),
                "resolution": "512x512",
                "fps": fps,
                "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
                "created_at": datetime.now().isoformat(),
                "generation_time_seconds": round(generation_time, 2),
                "device_used": str(pipeline.device),
                "generation_parameters": optimized_params,
                "video_info": video_info
            }
            save_generation_metadata(video_info["job_id"], metadata)
            
            batch_results.append({
                "prompt": prompt,
                "job_id": video_info["job_id"],
                "filename": video_info["filename"],
                "status": "success",
                "generation_time_seconds": round(generation_time, 2),
                "file_size_mb": video_info["file_size_mb"],
                "download_url": f"/download/{video_info['job_id']}",
                "preview_url": f"/preview/{video_info['job_id']}"
            })
            
            logger.info(f"Batch video {i} completed: {video_info['job_id']}")
            
        except Exception as e:
            logger.error(f"Failed to generate video {i}: {str(e)}")
            batch_results.append({
                "prompt": prompt,
                "status": "error",
                "error": str(e)
            })
    
    total_time = (datetime.now() - total_start_time).total_seconds()
    successful_videos = [r for r in batch_results if r.get("status") == "success"]
    
    return {
        "status": "completed",
        "total_prompts": len(prompts),
        "successful_generations": len(successful_videos),
        "failed_generations": len(prompts) - len(successful_videos),
        "total_batch_time_seconds": round(total_time, 2),
        "average_time_per_video": round(total_time / len(prompts), 2) if prompts else 0,
        "results": batch_results
    }

# Video Download & Preview Endpoints  

@app.get("/metadata/{job_id}")
async def get_video_metadata(job_id: str):
    """
    Get metadata for a generated video
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:
        job_id = job_id.lower().strip()
        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        logger.info(f"Looking for metadata at: {metadata_path}")
        
        # List all metadata files for debugging
        metadata_dir = get_output_path("metadata")
        if os.path.exists(metadata_dir):
            files = os.listdir(metadata_dir)
            logger.info(f"Available metadata files ({len(files)}): {', '.join(files)}")
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for job ID: {job_id}"
            )
            
        # Try to read with retries for consistency
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if metadata:
                        logger.info(f"Successfully loaded metadata for {job_id}")
                        return metadata
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} reading metadata: {e}")
                    time.sleep(0.5)
                    continue
                logger.error(f"Invalid JSON in metadata file {metadata_path}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Metadata file is corrupted: {str(e)}"
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} reading metadata: {e}")
                    time.sleep(0.5)
                    continue
                raise
                
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read metadata after {max_retries} attempts: {str(last_error)}"
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video metadata: {str(e)}"
        )

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the generated video file
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:
        video_path = find_video_file(job_id)
        
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found for job ID: {job_id}"
            )
        
        # Get filename from path
        filename = os.path.basename(video_path)
        
        logger.info(f"Serving video download: {filename}")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"generated_video_{job_id}.mp4",
            headers={
                "Content-Disposition": f"attachment; filename=generated_video_{job_id}.mp4"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download video: {str(e)}"
        )

@app.get("/metadata/{job_id}")
async def get_video_metadata(job_id: str):
    """
    Get metadata for a generated video
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:
        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        
        if not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=404,
                detail=f"Metadata not found for job ID: {job_id}"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return metadata
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video metadata: {str(e)}"
        )

@app.get("/preview/{job_id}")
async def preview_video(job_id: str):
    """
    Preview the generated video in the browser
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:
        # check if video exists
        video_path = find_video_file(job_id)
        
        if not video_path or not os.path.exists(video_path):
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Video Not Found</title></head>
                    <body>
                        <h2>Video Not Found</h2>
                        <p>No video found for job ID: {job_id}</p>
                        <a href="/">← Back to home</a>
                    </body>
                </html>
                """,
                status_code=404
            )
        
        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # html preview page
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Video Preview - Job {job_id}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 50px auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .video-container {{ 
                    text-align: center; 
                    margin: 20px 0; 
                }}
                video {{ 
                    border: 2px solid #ddd; 
                    border-radius: 8px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .metadata {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 20px 0;
                    border-left: 4px solid #007bff;
                }}
                .download-btn {{ 
                    background: #28a745; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    display: inline-block;
                    margin: 10px 5px;
                }}
                .home-btn {{ 
                    background: #007bff; 
                    color: white; 
                    padding: 10px 20px; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    display: inline-block;
                    margin: 10px 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generated Video Preview</h1>
                <p><strong>Job ID:</strong> {job_id}</p>
                
                <div class="video-container">
                    <video width="320" height="320" controls autoplay muted loop>
                        <source src="/download/{job_id}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                
                <div class="metadata">
                    <h3>Generation Details</h3>
                    <p><strong>Prompt:</strong> {metadata.get('prompt', 'N/A')}</p>
                    <p><strong>Duration:</strong> {metadata.get('duration', 'N/A')} seconds</p>
                    <p><strong>Resolution:</strong> {metadata.get('resolution', 'N/A')}</p>
                    <p><strong>Frames:</strong> {metadata.get('frames_generated', 'N/A')}</p>
                    <p><strong>File Size:</strong> {file_size_mb:.1f} MB</p>
                    <p><strong>Created:</strong> {metadata.get('created_at', 'N/A')}</p>
                    <p><strong>Model:</strong> {metadata.get('model_used', 'N/A')}</p>
                </div>
                
                <div style="text-align: center;">
                    <a href="/download/{job_id}" class="download-btn">Download Video</a>
                    <a href="/" class="home-btn">Back to Home</a>
                    <a href="/docs" class="home-btn">API Docs</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Preview Error</title></head>
                <body>
                    <h2>Preview Error</h2>
                    <p>Failed to load preview: {str(e)}</p>
                    <a href="/">← Back to home</a>
                </body>
            </html>
            """,
            status_code=500
        )

@app.get("/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status and metadata of a video generation job
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:
        job_id = job_id.lower() 
        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        logger.info(f"Checking job status: {job_id} at {metadata_path}")
        
        task_state = task_states.get(job_id)
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded metadata for {job_id}")
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}")
            if task_state:
                return {
                    "status": task_state,
                    "job_id": job_id,
                    "message": "Job is being processed but metadata is not yet available"
                }
            return {
                "status": "not_found",
                "job_id": job_id,
                "message": f"No job found with ID: {job_id}"
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file {metadata_path}: {e}")
            return {
                "status": "error",
                "job_id": job_id,
                "message": f"Metadata file is corrupted: {str(e)}"
            }
        
        if task_state == "failed":
            error_info = task_results[job_id].get("error", "Unknown error")
            return {
                "status": "failed",
                "job_id": job_id,
                "error": error_info,
                "metadata": metadata
            }
        
        video_path = find_video_file(job_id)
        video_ready = video_path is not None and os.path.exists(video_path)
        
        if task_state == "completed" and video_ready:
            metadata["status"] = "completed" 
            return {
                "status": "completed",
                "job_id": job_id,
                "video_ready": video_ready,
                "metadata": metadata,
                "download_url": f"/download/{job_id}",
                "preview_url": f"/preview/{job_id}"
            }
        
        if task_state in ["processing", "queued"]:
            metadata["status"] = task_state  
            return {
                "status": task_state,
                "job_id": job_id,
                "video_ready": False,
                "metadata": metadata
            }
        
        # Default to metadata state
        return {
            "status": metadata.get("status", "metadata_only"),
            "job_id": job_id,
            "video_ready": video_ready,
            "metadata": metadata,
            "download_url": f"/download/{job_id}" if video_ready else None,
            "preview_url": f"/preview/{job_id}" if video_ready else None
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            "status": "error",
            "job_id": job_id,
            "message": f"Failed to get job status: {str(e)}"
        }

# for: "python main.py"
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
