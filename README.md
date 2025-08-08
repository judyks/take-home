# Video Generation API

This API generates videos from text prompts using the `Lightricks/LTX-Video-0.9.7-distilled` model. It's built with FastAPI and runs in a Docker container, optimized for NVIDIA H100 GPUs.

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

## API Endpoints

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

