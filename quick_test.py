#!/usr/bin/env python3
"""
Quick API Test - Test the main functionality of the Video Generation API
"""

import requests
import time
import json
import sys

def test_api():
    base_url = "http://localhost:8000"
    session = requests.Session()
    
    print("Video Generation API Quick Test")
    print("=" * 40)
    
    # Test 1: Health Check
    print("\n1. Health Check...")
    try:
        response = session.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   API is healthy")
            print(f"   GPU Available: {data.get('gpu_available')}")
        else:
            print(f"   Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Cannot connect to API: {e}")
        return False
    
    # Test 2: Model Status  
    print("\n2. Model Status...")
    try:
        response = session.get(f"{base_url}/model-status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "loaded":
                print(f"   Model loaded: {data.get('model_id')}")
            else:
                print(f"   Model not loaded: {data.get('status')}")
                return False
        else:
            print(f"   Model status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Error checking model: {e}")
        return False
        
    # Test 3: Video Generation
    print("\n3. Video Generation...")
    try:
        params = {
            "prompt": "A simple bouncing ball",
            "duration": 3
        }
        response = session.post(f"{base_url}/generate", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                job_id = data.get("job_id")
                print(f"   Video generation started: {job_id}")
            else:
                print(f"   Generation failed: {data.get('message')}")
                return False
        else:
            print(f"   Generation request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Error generating video: {e}")
        return False
        
    # Test 4: Job Status
    print("\n4. Job Status...")
    try:
        response = session.get(f"{base_url}/status/{job_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            print(f"   Job status: {status}")
            
            if status == "completed":
                print(f"   Video ready for download!")
                return True
            elif status == "processing":
                print(f"   Video is still processing...")
                return True
            elif status == "failed":
                print(f"   Video generation failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"   Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   Error checking status: {e}")
        return False
        
    return True

def main():
    try:
        success = test_api()
        print("\n" + "=" * 40)
        if success:
            print("All tests passed! API is working correctly.")
            print("\nNext steps:")
            print("- Check /docs for full API documentation")
            print("- Use /generate endpoint to create videos")
            print("- Monitor jobs with /status/{job_id}")
            print("- Download videos with /download/{job_id}")
        else:
            print("Some tests failed. Check the API server.")
            
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
