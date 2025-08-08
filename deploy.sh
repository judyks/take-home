#!/bin/bash

set -e

echo "Starting Video Generation API Deployment"
echo "========================================"

if [ ! -f "docker-compose.yml" ]; then
    echo "Error: docker-compose.yml not found. Are you in the project root?"
    exit 1
fi

check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "Docker is running"
}

check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "No NVIDIA GPU detected (nvidia-smi not found)"
        echo "   The API will still work but will use CPU mode"
    fi
}

build_image() {
    echo ""
    echo "Building Docker image..."
    echo "This may take 5-10 minutes for the first build..."
    
    docker-compose build --no-cache
    
    if [ $? -eq 0 ]; then
        echo "Docker image built successfully"
    else
        echo "Docker build failed"
        exit 1
    fi
}

start_service() {
    echo ""
    echo "Starting Video Generation API..."
    
    docker-compose down 2>/dev/null || true
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        echo "Service started successfully"
    else
        echo "Failed to start service"
        exit 1
    fi
}

wait_for_service() {
    echo ""
    echo "Waiting for API to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            echo "API is ready!"
            break
        fi
        
        echo "   Attempt $attempt/$max_attempts - API not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        echo "API failed to start within expected time"
        echo "Checking logs..."
        docker-compose logs --tail=20
        exit 1
    fi
}

run_tests() {
    echo ""
    echo "Running API tests..."
    
    pip install requests >/dev/null 2>&1 || pip3 install requests >/dev/null 2>&1 || true
    
    if [ -f "tests/quick_test.py" ]; then
        python3 tests/quick_test.py
    else
        echo "tests/quick_test.py not found, skipping automated tests"
    fi
}

show_status() {
    echo ""
    echo "Deployment Complete!"
    echo "==================="
    
    echo "Container Status:"
    docker-compose ps
    
    echo ""
    echo "Access Points:"
    echo "API Homepage: http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/health"
    echo "Model Status: http://localhost:8000/model-status"
    
    echo ""
    echo "Quick Test Commands:"
    echo "Load model: curl http://localhost:8000/test-model-loading"
    echo "Generate video: curl -X POST 'http://localhost:8000/generate?prompt=A%20sunset&duration=3'"
    
    echo ""
    echo "Useful Commands:"
    echo "View logs: docker-compose logs -f"
    echo "Stop service: docker-compose down"
    echo "Rebuild: docker-compose build --no-cache"
    echo "Shell access: docker-compose exec video-api bash"
    
    echo ""
    echo "Output Files:"
    echo "Videos: ./outputs/videos/"
    echo "Metadata: ./outputs/metadata/"
    echo "Logs: ./logs/"
}

main() {
    check_docker
    check_gpu
    build_image
    start_service
    wait_for_service
    run_tests
    show_status
}

if main; then
    echo ""
    echo "All systems ready! Your Video Generation API is live!"
else
    echo ""
    echo "Deployment failed. Check the logs above for details."
    echo "For troubleshooting, run: docker-compose logs"
    exit 1
fi
