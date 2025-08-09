#!/usr/bin/env python3
"""
Simple test script for the video generation API
"""

import requests
import json
import time
import random

def test_api():
    api_url = "http://localhost:8000"  # Use the known working port
    
    # Random prompts to choose from
    prompts = [
        "A magical butterfly dancing in a field of glowing flowers at twilight",
        "A robot chef making pancakes in a futuristic kitchen",
        "Ocean waves crashing against ancient castle ruins under a starry night",
        "A red balloon floating through a vibrant rainbow after rain",
        "A curious cat exploring a garden filled with colorful butterflies"
    ]
    
    selected_prompt = random.choice(prompts)
    
    print("🎬 Video Generation API Test")
    print("=" * 50)
    print(f"Selected prompt: {selected_prompt}")
    print("=" * 50)
    
    # Check health
    try:
        print("📊 Checking API health...")
        response = requests.get(f"{api_url}/health", timeout=10)
        health_data = response.json()
        print(f"Status: {health_data['status']}")
        print(f"GPU Available: {health_data['gpu_available']}")
        if health_data.get('gpu_info'):
            print(f"GPU: {health_data['gpu_info']['device_name']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Generate video
    print(f"\n🎥 Generating video...")
    print("This may take several minutes, especially on first run...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{api_url}/generate",
            params={
                "prompt": selected_prompt,
                "duration": 6,  # 6 seconds instead of 3
                "seed": random.randint(1, 100000)
            },
            timeout=600  # 10 minute timeout
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success! Video generated in {generation_time:.1f} seconds")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Try to extract fields that might exist
            job_id = result.get('job_id') or result.get('video_id')
            filename = result.get('filename')
            
            if job_id:
                print(f"Job/Video ID: {job_id}")
            if filename:
                print(f"Filename: {filename}")
                print(f"Download URL: {api_url}/download/{job_id}")
                print(f"Preview URL: {api_url}/preview/{job_id}")
        else:
            print(f"\n❌ Generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"\n⏰ Timeout after 10 minutes - this might be normal for first-time model loading")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_api()
