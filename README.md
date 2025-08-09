# Video Generation API

This API generates videos from text prompts using the `Lightricks/LTX-Video-0.9.7-distilled` model. It's built with FastAPI and runs in a Docker container, optimized for NVIDI├── src
│   └── main.py
└── tests
    ├── quick_test.py
    └── test_video_generation.py## Project Structure
```
.
├── Dockerfile                    # Container definition
├── README.md                     # This documentation
├── demo.py                       # End-to-end demo script
├── deploy.sh                     # One-command deployment
├── docker-compose.yml            # Service orchestration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment configuration template
├── configs/
│   └── settings.py              # Application configuration
├── src/
│   └── main.py                  # Main API application
└── tests/
    ├── quick_test.py            # Basic functionality tests
    └── test_video_generation.py # Comprehensive tests
```leshooting

### Common Issues

#### API Not Starting
- **Symptom**: `curl: (7) Failed to connect to localhost port 8000`
- **Solution**: Check if Docker is running: `docker-compose ps`
- **Alternative**: Check logs: `docker-compose logs video-api`

#### CUDA Errors on Start
- **Symptom**: `RuntimeError: CUDA out of memory` or similar
- **Solution**: This usually indicates a mismatch between the Docker runtime and NVIDIA drivers
- **Fix**: Try rebuilding the image: `docker-compose build --no-cache`
- **Alternative**: Ensure NVIDIA Docker runtime is installed

#### Model Loading Fails
- **Symptom**: Model loading endpoint returns errors
- **Solution**: Check internet connection for HuggingFace model download
- **Alternative**: Verify GPU memory availability: `nvidia-smi`

#### Out of Memory (OOM)
- **Symptom**: Generation fails with memory errors
- **Solution**: Reduce video duration or resolution
- **Alternative**: Restart container to clear GPU memory: `docker-compose restart`

#### Video Download Fails
- **Symptom**: 404 errors when downloading videos
- **Solution**: Check if video generation completed: `curl http://localhost:8000/status/{job_id}`
- **Alternative**: Check storage permissions and disk space

### Performance Issues

#### Slow Generation
- **Cause**: CPU-only mode or insufficient GPU memory
- **Solution**: Verify GPU is being used: `nvidia-smi`
- **Optimization**: Use shorter prompts and reduce inference steps

#### Storage Full
- **Symptom**: Cannot save videos
- **Solution**: Clean old videos manually or adjust `CLEANUP_AFTER_DAYS`
- **Prevention**: Monitor disk usage with `df -h`

### Debugging Commands

```bash
# Check container status
docker-compose ps

# View real-time logs
docker-compose logs -f video-api

# Check GPU usage
nvidia-smi

# Test API health
curl http://localhost:8000/health

# Get shell access to container
docker-compose exec video-api bash

# Check disk space
df -h

# Check model loading status
curl http://localhost:8000/model-status
```

### Getting Help

1. **Check Logs**: Always start with `docker-compose logs video-api`
2. **Test Health**: Verify basic functionality with `/health` endpoint
3. **GPU Status**: Ensure GPU is accessible with `nvidia-smi`
4. **Model Status**: Check model loading with `/model-status`
5. **Storage**: Verify disk space and permissions

---

## Summary

This Video Generation API provides:
- **Enterprise-ready deployment** with Docker and orchestration
- **Comprehensive monitoring** with health checks and logging
- **Production configuration** with environment variables
- **Complete testing suite** for validation
- **Detailed documentation** for operation and troubleshooting

The API is designed for reliability, scalability, and ease of operation in production environments.

## Quick Start

### One-Command Deployment
For a quick and easy setup, just run the deployment script. This will build the Docker image, start the service, and run tests.

```bash
./deploy.sh
```

### Manual Deployment
If you prefer to run the steps manually:

```bash
# Build and start the Docker container in the background
docker-compose up --build -d

# Check the health of the API
curl http://localhost:8000/health

# Generate a video from a text prompt
curl -X POST "http://localhost:8000/generate?prompt=A%20cat%20walking"
```

## System Architecture

### Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │───▶│   FastAPI App    │───▶│  LTX-Video Model│
│  (Browser/API)  │    │   (main.py)      │    │   (H100 GPU)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  File Storage    │
                       │ (videos/metadata)│
                       └──────────────────┘
```

### Components
- **FastAPI Application**: RESTful API server with automatic OpenAPI documentation
- **LTX-Video Model**: HuggingFace diffusion model for text-to-video generation
- **Storage System**: Organized file storage for videos and metadata
- **Docker Container**: Containerized deployment with GPU support
- **Web Interface**: Built-in gallery and preview system

## API Documentation

The base URL for all endpoints is `http://localhost:8000`.

### Core
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Homepage with an API overview. |
| `/docs` | GET | Interactive API documentation (Swagger UI). |
| `/health` | GET | Check system health and GPU status. |
| `/model-status` | GET | Check if the video generation model is loaded. |

### Video Generation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate a video from a prompt. |

### Video Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/download/{job_id}` | GET | Download a generated video. |
| `/preview/{job_id}` | GET | Preview a video in the browser. |
| `/status/{job_id}` | GET | Get the status and metadata for a generation job. |
| `/gallery` | GET | Web interface showing all generated videos. |

### Examples

#### Generate a Video
```bash
curl -X POST "http://localhost:8000/generate" \
  -d "prompt=A beautiful sunset over the ocean" \
  -d "duration=5"
```

#### Check Generation Status
```bash
curl "http://localhost:8000/status/abc123def"
```

#### Download Video
```bash
curl -o my_video.mp4 "http://localhost:8000/download/abc123def"
```

## 🛠️ Configuration

### Environment Variables
The API supports configuration via environment variables:

```bash
# Model Settings
MODEL_NAME=Lightricks/LTX-Video-0.9.7-distilled
MODEL_DEVICE=auto  # auto, cuda, cpu
MAX_VIDEO_DURATION=10
DEFAULT_VIDEO_DURATION=3

# Storage Settings
OUTPUT_DIR=./outputs
MAX_STORAGE_GB=100
CLEANUP_AFTER_DAYS=7

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Security (optional)
ENABLE_RATE_LIMIT=false
API_KEY_REQUIRED=false
```

### Docker Environment File
Create a `.env` file for Docker Compose:

```bash
# .env file
CUDA_VISIBLE_DEVICES=0
MODEL_NAME=Lightricks/LTX-Video-0.9.7-distilled
MAX_VIDEO_DURATION=10
LOG_LEVEL=INFO
```

## Testing

### Demo Script
Run the complete end-to-end demo:
```bash
python demo.py
```

### Quick Test
Run basic functionality tests:
```bash
python tests/quick_test.py
```

### Comprehensive Test
Run full end-to-end tests including video generation:
```bash
python tests/test_video_generation.py
```

### Manual Testing
```bash
# Test the API manually
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/generate?prompt=Test%20video&duration=3"
```

## Useful Commands

### Interacting with the API
```bash
# Get API health status
curl http://localhost:8000/health

# Load the model into memory
curl http://localhost:8000/test-model-loading

# Generate a video
curl -X POST "http://localhost:8000/generate?prompt=A%20beautiful%20sunset&duration=5"

# Check the status of your video generation job
curl http://localhost:8000/status/{job_id}

# Download your video
curl -o video.mp4 http://localhost:8000/download/{job_id}
```

### Managing the Service
```bash
# View real-time logs
docker-compose logs -f

# Stop the service
docker-compose down

# Rebuild the Docker image from scratch
docker-compose build --no-cache

# Get a shell inside the running container
docker-compose exec video-api bash
```

## Project Structure
```
.
├── Dockerfile
├── README.md
├── configs
│   └── settings.py
├── deploy.sh
├── docker-compose.yml
├── requirements.txt
├── requirements_h100.txt
├── src
│   └── main.py
└── tests
    ├── quick_test.py
    └── test_video_generation.py
```

## Troubleshooting

*   **CUDA Errors on Start**: This usually indicates a mismatch between the Docker runtime and the NVIDIA drivers. Try rebuilding the image with `docker-compose build --no-cache`.
*   **Out of Memory (OOM)**: If you're running out of GPU memory, try reducing the `duration` or resolution of the generated video.
*   **API Not Responding**: Make sure the Docker container is running (`docker-compose ps`) and that nothing else is using port 8000.

