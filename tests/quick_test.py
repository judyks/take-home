#!/usr/bin/env python3
"""
Quick API Test Script
Tests basic functionality of the Video Generation API
"""

import requests
import time
import json
import sys
import os

def test_api(base_url="http://localhost:8000"):
    """Run basic API tests"""
    
    print("Video Generation API - Quick Test")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("   Health check passed")
            health_data = response.json()
            print(f"   GPU Available: {health_data.get('gpu_available', False)}")
            tests_passed += 1
        else:
            print(f"   Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   Health check error: {e}")
    
    # Test 2: Root Endpoint
    print("\n2. Testing root endpoint...")
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("   Root endpoint works")
            tests_passed += 1
        else:
            print(f"   Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   Root endpoint error: {e}")
    
    # Test 3: API Documentation
    print("\n3. Testing API documentation...")
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("   API docs accessible")
            tests_passed += 1
        else:
            print(f"   API docs failed: {response.status_code}")
    except Exception as e:
        print(f"   API docs error: {e}")
    
    # Test 4: Model Status
    print("\n4. Testing model status...")
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/model-status", timeout=10)
        if response.status_code == 200:
            print("   ✅ Model status endpoint works")
            status_data = response.json()
            print(f"   📋 Model Status: {status_data.get('status', 'unknown')}")
            tests_passed += 1
        else:
            print(f"   Model status failed: {response.status_code}")
    except Exception as e:
        print(f"   Model status error: {e}")
    
    # Test 5: Gallery Endpoint
    print("\n5. Testing gallery endpoint...")
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/gallery", timeout=10)
        if response.status_code == 200:
            print("   Gallery endpoint works")
            tests_passed += 1
        else:
            print(f"   Gallery failed: {response.status_code}")
    except Exception as e:
        print(f"   Gallery error: {e}")
    
    # Results Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("All tests passed! API is working correctly.")
        print(f"\nAccess the API:")
        print(f"   • Homepage: {base_url}")
        print(f"   • API Docs: {base_url}/docs")
        print(f"   • Gallery: {base_url}/gallery")
        print(f"   • Health: {base_url}/health")
        
        print(f"\nGenerate a video:")
        print(f"   curl -X POST '{base_url}/generate?prompt=A%20cat%20walking&duration=3'")
        return True
    else:
        print("Some tests failed. Check the API deployment.")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        print("Waiting 30 seconds for server to start...")
        time.sleep(30)
    
    # Allow custom base URL
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    if len(sys.argv) > 1 and sys.argv[-1].startswith("http"):
        base_url = sys.argv[-1]
    
    success = test_api(base_url)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
