#!/usr/bin/env python3
"""
Test Video Generation API with a Random Prompt
"""

import requests
import time
import json
import os
import random

def test_with_random_prompt():
    """Test the API with a randomly generated prompt"""
    
    # List of creative random prompts
    random_prompts = [
        "A magical butterfly dancing in a field of glowing flowers at twilight",
        "A robot chef making pancakes in a futuristic kitchen",
        "Ocean waves crashing against ancient castle ruins under a starry night",
        "A red balloon floating through a vibrant rainbow after rain",
        "A curious cat exploring a garden filled with colorful butterflies",
        "Fireflies illuminating a dark forest path at midnight",
        "A vintage train traveling through mountains covered in autumn leaves",
        "Dolphins jumping gracefully through crystal clear blue water",
        "A hot air balloon drifting over a city skyline at sunset",
        "Snowflakes falling gently on a cozy cabin in the woods"
    ]
    
    # Select a random prompt
    selected_prompt = random.choice(random_prompts)
    
    print("=" * 60)
    print("Video Generation API - Random Prompt Test")
    print("=" * 60)
    print(f"Selected Random Prompt: {selected_prompt}")
    print("-" * 60)
    
    # Try different possible ports/URLs
    possible_urls = [
        "http://localhost:8000",
        "http://localhost:8001", 
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001"
    ]
    
    api_url = None
    
    # Find working API endpoint
    for url in possible_urls:
        try:
            print(f"Trying API at {url}...")
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                api_url = url
                print(f"✓ API found at {url}")
                break
        except requests.exceptions.RequestException:
            print(f"✗ No API at {url}")
            continue
    
    if not api_url:
        print("\n❌ No running API found. Starting local API...")
        # Try to start the API directly
        import sys
        import subprocess
        import threading
        
        def start_api():
            try:
                subprocess.run([
                    sys.executable, "-c", 
                    "import sys; sys.path.append('src'); from main import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8002)"
                ], cwd="/home/user/video-generation-api/take-home")
            except Exception as e:
                print(f"Failed to start API: {e}")
        
        # Start API in background thread
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        
        # Wait and try to connect
        print("Waiting for API to start...")
        time.sleep(5)
        
        try:
            response = requests.get("http://127.0.0.1:8002/health", timeout=5)
            if response.status_code == 200:
                api_url = "http://127.0.0.1:8002"
                print("✓ API started successfully")
            else:
                raise Exception("API not responding")
        except:
            print("❌ Could not start API. Please run manually:")
            print("cd /home/user/video-generation-api/take-home")
            print("python src/main.py")
            return
    
    # Test the API
    print(f"\n🚀 Testing video generation with prompt:")
    print(f"'{selected_prompt}'")
    
    try:
        # Check health first
        health_response = requests.get(f"{api_url}/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"\n📊 API Status: {health_data.get('status')}")
            print(f"🔧 GPU Available: {health_data.get('gpu_available')}")
            
            if health_data.get('gpu_info'):
                gpu_info = health_data['gpu_info']
                print(f"🎮 GPU: {gpu_info.get('device_name')}")
        
        # Generate video
        print(f"\n🎬 Generating video (this may take a while)...")
        
        generation_data = {
            "prompt": selected_prompt,
            "duration": 3,  # 3 seconds
            "seed": random.randint(1, 1000000)
        }
        
        start_time = time.time()
        response = requests.post(
            f"{api_url}/generate",
            data=generation_data,
            timeout=300  # 5 minute timeout
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Video generated successfully!")
            print(f"⏱️  Generation time: {generation_time:.1f} seconds")
            print(f"📁 Video ID: {result.get('video_id')}")
            print(f"📄 Filename: {result.get('filename')}")
            print(f"📊 Prompt used: {result.get('prompt')}")
            
            # Try to get video info
            if result.get('video_id'):
                info_response = requests.get(f"{api_url}/video/{result['video_id']}/info")
                if info_response.status_code == 200:
                    info = info_response.json()
                    print(f"🎞️  Duration: {info.get('duration_seconds', 'unknown')} seconds")
                    print(f"📐 Resolution: {info.get('width', '?')}x{info.get('height', '?')}")
                    print(f"🎯 FPS: {info.get('fps', 'unknown')}")
            
            print(f"\n📥 You can download the video at: {api_url}/video/{result.get('video_id')}/download")
            print(f"👀 Or preview it at: {api_url}/video/{result.get('video_id')}/preview")
            
        else:
            print(f"\n❌ Video generation failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Error response: {response.text}")
                
    except requests.exceptions.Timeout:
        print(f"\n⏰ Request timed out - video generation is taking longer than expected")
        print("This might be normal for the first generation as the model needs to load")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_with_random_prompt()
