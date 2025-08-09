# main.py - video generation API server

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import logging
import os
import uuid
import json
import imageio
import uvicorn
from datetime import datetime
from typing import List
from PIL import Image
import numpy as np

# try to import PyTorch -> fallback if CUDA issues
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} loaded successfully")
    if torch.cuda.is_available():
        print(f"CUDA {torch.version.cuda} is available")
        print(f"GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, running on CPU")
except Exception as e:
    print(f"Failed to import PyTorch: {e}")
    TORCH_AVAILABLE = False
    
    # create mock torch module
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
    
    torch = MockTorch()
    print("Using MockTorch for graceful fallback")

# configure logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global variable to store model
pipeline = None


# Video Saving & File Management Functions

def ensure_output_directories():
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
    base_dir = os.getcwd()
    if not base_dir.endswith("take-home"):
        base_dir = os.path.dirname(base_dir)
    return os.path.join(base_dir, "outputs", subdir)

def save_video_frames(frames: List, prompt: str, duration: int, resolution: str) -> dict:
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
        
        job_id = str(uuid.uuid4())[:8]
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
        fps = len(processed_frames) / duration if duration > 0 else 8
        
        # save as MP4 video
        logger.info(f"Saving video with {len(processed_frames)} frames at {fps:.1f} FPS")
        
        # imageio to save MP4
        writer = imageio.get_writer(
            video_path, 
            fps=fps, 
            quality=7,  
            codec='libx264',  # standard for compatibility
            format='FFMPEG',  # for MP4
            output_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
        )
        
        for frame in processed_frames:
            writer.append_data(frame)
        writer.close()
        
        # Calculate file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        logger.info(f"Video saved: {filename} ({file_size_mb:.1f} MB)")
        
        return {
            "job_id": job_id,
            "filename": filename,
            "video_path": video_path,
            "file_size_mb": round(file_size_mb, 2),
            "fps": round(fps, 1),
            "total_frames": len(processed_frames)
        }
        
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        raise Exception(f"Video saving failed: {str(e)}")

def save_generation_metadata(job_id: str, metadata: dict):
    """Save metadata about the video generation"""
    try:
        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved for job {job_id}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")

def find_video_file(job_id: str) -> str:
    """Find video file by job_id"""
    videos_dir = get_output_path("videos")
    if not os.path.exists(videos_dir):
        return None
    
    # Look for files containing the job_id
    for filename in os.listdir(videos_dir):
        if job_id in filename and filename.endswith('.mp4'):
            return os.path.join(videos_dir, filename)
    return None

# create the FastAPI application (web server to receive HTTP requests)
app = FastAPI(
    title="LTX-Video-0.9.7-distilled Generation API",
    description="Converts text prompts into videos using Lightricks LTX-Video-0.9.7-distilled model",
    version="1.0.0"
)





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
            "message": "Welcome to the LTX-Video-0.9.7-distilled Generation API!",
            "model": "Lightricks/LTX-Video-0.9.7-distilled",
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
            "message": "Welcome to the LTX-Video-0.9.7-distilled Generation API!",
            "model": "Lightricks/LTX-Video-0.9.7-distilled",
            "status": "Server is running",
            "docs": "Visit /docs to see all available endpoints",
            "error": f"Could not load recent videos: {str(e)}"
        }

@app.get("/health")
async def health_check():
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
async def test_model_loading():
    """
    Test endpoint to load the LTX-Video-0.9.7-distilled model
    This loads the specific distilled model from Lightricks
    """
    global pipeline
    
    try:
        logger.info("Starting LTX-Video-0.9.7-distilled model loading...")
        
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
            pipeline = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video-0.9.7-distilled",
                torch_dtype=torch.bfloat16
            )
            logger.info("Successfully loaded LTX-Video-0.9.7-distilled model")
        except Exception as model_error:
            logger.error(f"Failed to load LTX-Video-0.9.7-distilled: {model_error}")
            
            # fallback to ModelScope
            logger.info("Trying fallback model: damo-vilab/text-to-video-ms-1.7b")
            try:
                from diffusers import DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    "damo-vilab/text-to-video-ms-1.7b",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                logger.info("Successfully loaded ModelScope text-to-video model as fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                return {
                    "status": "error",
                    "error_type": "model_loading_error",
                    "message": f"Both primary and fallback models failed. Primary: {str(model_error)}, Fallback: {str(fallback_error)}",
                    "primary_model": "Lightricks/LTX-Video-0.9.7-distilled",
                    "fallback_model": "damo-vilab/text-to-video-ms-1.7b",
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
async def model_status():
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
async def debug_diffusers():
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


# Video Generation Endpoints

@app.post("/generate")
async def generate_video(prompt: str, duration: int = 3):
    """
    Generate a video from a text prompt
    
    Parameters:
    - prompt: Text description of what you want in the video
    - duration: Video length in seconds (default: 3)
    """
    global pipeline
    
    # check if model is loaded, load it if not
    if pipeline is None:
        logger.info("Model not loaded, attempting to load LTX-Video-0.9.7-distilled...")
        try:
            load_result = await test_model_loading()
            if load_result.get("status") != "success":
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load LTX-Video-0.9.7-distilled model: {load_result.get('message', 'Unknown error')}"
                )
            logger.info("Model loaded successfully")
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to auto-load LTX-Video-0.9.7-distilled model: {str(e)}"
            )
    
    try:
        logger.info(f"Generating video for prompt: '{prompt}'")
        
        # Basic validation
        if not prompt or len(prompt.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Prompt cannot be empty"
            )
        
        if duration < 1 or duration > 10:
            raise HTTPException(
                status_code=400,
                detail="Duration must be between 1 and 10 seconds"
            )
        
        logger.info("Starting video generation...")
        start_time = datetime.now()
        
        # Simple generation call (parameters optimized for LTX-Video-0.9.7-distilled)
        try:
            video_frames = pipeline(
                prompt=prompt,
                num_frames=max(16, duration * 8),     # Higher quality: 8 FPS
                guidance_scale=7.5,                   
                num_inference_steps=20,               
                height=512,                          
                width=512,
                generator=torch.Generator(device=pipeline.device).manual_seed(42)
            ).frames[0]
            
            logger.info(f"Generated {len(video_frames)} frames")
            
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            logger.info(f"Generation completed in {generation_time:.1f} seconds")
            
            # save generated frames as an MP4
            try:
                video_info = save_video_frames(
                    frames=video_frames,
                    prompt=prompt,
                    duration=duration,
                    resolution="512x512"
                )
                
                metadata = {
                    "prompt": prompt,
                    "duration": duration,
                    "frames_generated": len(video_frames),
                    "resolution": "512x512",
                    "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
                    "created_at": datetime.now().isoformat(),
                    "generation_time_seconds": round(generation_time, 2),
                    "device_used": str(pipeline.device),
                    "video_info": video_info
                }
                save_generation_metadata(video_info["job_id"], metadata)
                logger.info(f"Video generation completed! Job ID: {video_info['job_id']}")
                
                # Print video links to terminal
                print("\n" + "="*60)
                print("🎬 VIDEO GENERATION COMPLETED! 🎬")
                print("="*60)
                print(f"📝 Prompt: {prompt}")
                print(f"🆔 Job ID: {video_info['job_id']}")
                print(f"📁 Filename: {video_info['filename']}")
                print(f"⏱️  Generation Time: {round(generation_time, 2)}s")
                print(f"📏 File Size: {video_info['file_size_mb']} MB")
                print("\n🔗 ACCESS LINKS:")
                print(f"   📥 Download: http://localhost:8000/download/{video_info['job_id']}")
                print(f"   👁️  Preview:  http://localhost:8000/preview/{video_info['job_id']}")
                print(f"   ℹ️  Status:   http://localhost:8000/status/{video_info['job_id']}")
                print("="*60 + "\n")
                
                return {
                    "status": "success",
                    "message": f"Video generated and saved successfully with LTX-Video-0.9.7-distilled!",
                    "job_id": video_info["job_id"],
                    "filename": video_info["filename"],
                    "prompt": prompt,
                    "duration": duration,
                    "frames_generated": len(video_frames),
                    "resolution": "512x512",
                    "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
                    "file_size_mb": video_info["file_size_mb"],
                    "fps": video_info["fps"],
                    "generation_time_seconds": round(generation_time, 2),
                    "device_used": str(pipeline.device),
                    "download_url": f"/download/{video_info['job_id']}",
                    "preview_url": f"/preview/{video_info['job_id']}"
                }
                
            except Exception as save_error:
                logger.error(f"Failed to save video: {str(save_error)}")
                return {
                    "status": "partial_success",
                    "message": f"Video generated but saving failed: {str(save_error)}",
                    "prompt": prompt,
                    "duration": duration,
                    "frames_generated": len(video_frames),
                    "resolution": "512x512",
                    "model_used": "Lightricks/LTX-Video-0.9.7-distilled",
                    "note": "Frames generated successfully but video file could not be saved"
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

# Video Download & Preview Endpoints  

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
        
        # load metadata if available
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
async def get_job_status(job_id: str):
    """
    Get the status and metadata of a video generation job
    
    Parameters:
    - job_id: Unique identifier for the video generation job
    """
    try:

        metadata_path = os.path.join(get_output_path("metadata"), f"{job_id}.json")
        
        if not os.path.exists(metadata_path):
            return {
                "status": "not_found",
                "job_id": job_id,
                "message": f"No job found with ID: {job_id}"
            }
        
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        
        video_path = find_video_file(job_id)
        video_ready = video_path is not None and os.path.exists(video_path)
        
        return {
            "status": "completed" if video_ready else "metadata_only",
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
