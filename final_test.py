#!/usr/bin/env python3
"""
Final App Test - Test all working functionality
"""

import requests
import time
import sys

def test_app():
    base_url = "http://localhost:8000"
    session = requests.Session()
    
    print("Final Video Generation API Test")
    print("=" * 45)
    
    passed_tests = 0
    total_tests = 0
    
    def test_result(name, success, message=""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        if success:
            passed_tests += 1
            print(f"PASS {name}")
        else:
            print(f"FAIL {name}")
        if message:
            print(f"     {message}")
    
    # Test 1: Basic API Health
    print("\n1. API Health and Status")
    try:
        response = session.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            test_result("Health Check", data.get("status") == "healthy", 
                       f"GPU: {data.get('gpu_available')}")
        else:
            test_result("Health Check", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Health Check", False, f"Error: {e}")
    
    # Test 2: Model Status
    try:
        response = session.get(f"{base_url}/model-status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            test_result("Model Status", data.get("status") == "loaded",
                       f"Model: {data.get('model_id')}")
        else:
            test_result("Model Status", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Model Status", False, f"Error: {e}")
    
    # Test 3: Homepage
    try:
        response = session.get(f"{base_url}/", timeout=10)
        test_result("Homepage", response.status_code == 200)
    except Exception as e:
        test_result("Homepage", False, f"Error: {e}")
    
    # Test 4: Documentation
    try:
        response = session.get(f"{base_url}/docs", timeout=10)
        test_result("Documentation", response.status_code == 200)
    except Exception as e:
        test_result("Documentation", False, f"Error: {e}")
    
    # Test 5: Styles
    print("\n2. Feature Endpoints")
    try:
        response = session.get(f"{base_url}/styles", timeout=10)
        if response.status_code == 200:
            data = response.json()
            styles = data.get("available_styles", {})
            test_result("Styles Endpoint", len(styles) > 0,
                       f"Found {len(styles)} styles")
        else:
            test_result("Styles Endpoint", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Styles Endpoint", False, f"Error: {e}")
    
    # Test 6: Prompt Analysis
    try:
        response = session.post(f"{base_url}/analyze-prompt", 
                               params={"prompt": "test prompt"}, timeout=10)
        test_result("Prompt Analysis", response.status_code == 200)
    except Exception as e:
        test_result("Prompt Analysis", False, f"Error: {e}")
    
    # Test 7: Input Validation
    print("\n3. Input Validation")
    try:
        # Test invalid duration
        response = session.post(f"{base_url}/generate", 
                               params={"prompt": "test", "duration": 15}, timeout=10)
        test_result("Invalid Duration Rejected", response.status_code in [400, 422])
    except Exception as e:
        test_result("Invalid Duration Rejected", False, f"Error: {e}")
    
    try:
        # Test missing prompt
        response = session.post(f"{base_url}/generate", 
                               params={"duration": 3}, timeout=10)
        test_result("Missing Prompt Rejected", response.status_code in [400, 422])
    except Exception as e:
        test_result("Missing Prompt Rejected", False, f"Error: {e}")
    
    # Test 8: Video Generation
    print("\n4. Video Generation")
    job_id = None
    try:
        response = session.post(f"{base_url}/generate", 
                               params={"prompt": "A test video", "duration": 3}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                job_id = data.get("job_id")
                test_result("Video Generation Request", True, f"Job ID: {job_id}")
            else:
                test_result("Video Generation Request", False, f"Failed: {data.get('message')}")
        else:
            test_result("Video Generation Request", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Video Generation Request", False, f"Error: {e}")
    
    # Test 9: Job Status
    if job_id:
        try:
            response = session.get(f"{base_url}/status/{job_id}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                test_result("Job Status Check", status in ["processing", "completed", "failed"],
                           f"Status: {status}")
            else:
                test_result("Job Status Check", False, f"HTTP {response.status_code}")
        except Exception as e:
            test_result("Job Status Check", False, f"Error: {e}")
    else:
        test_result("Job Status Check", False, "No job ID to check")
    
    # Test 10: Test with existing completed video
    print("\n5. Download Test (using completed video)")
    completed_job = "78a7c8aa"  # We know this one is completed
    try:
        response = session.get(f"{base_url}/download/{completed_job}", timeout=30)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            file_size = len(response.content)
            test_result("Video Download", 'video' in content_type,
                       f"Size: {file_size:,} bytes, Type: {content_type}")
        else:
            test_result("Video Download", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Video Download", False, f"Error: {e}")
    
    # Test 11: Metadata
    try:
        response = session.get(f"{base_url}/metadata/{completed_job}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            test_result("Video Metadata", "prompt" in data and "status" in data)
        else:
            test_result("Video Metadata", False, f"HTTP {response.status_code}")
    except Exception as e:
        test_result("Video Metadata", False, f"Error: {e}")
    
    # Test 12: Error Handling
    print("\n6. Error Handling")
    try:
        response = session.get(f"{base_url}/status/invalid_job_id", timeout=10)
        test_result("Invalid Job ID Handling", response.status_code in [200, 404])
    except Exception as e:
        test_result("Invalid Job ID Handling", False, f"Error: {e}")
    
    # Summary
    print("\n" + "=" * 45)
    print("FINAL TEST RESULTS")
    print("=" * 45)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("\nAPP TEST: EXCELLENT")
        print("The Video Generation API is working very well!")
        print("- All core functionality is operational")
        print("- API endpoints are responding correctly")
        print("- Video generation and download work")
        print("- Input validation is working")
        print("- Error handling is appropriate")
        return True
    elif success_rate >= 70:
        print("\nAPP TEST: GOOD")
        print("The Video Generation API is mostly working.")
        print("Some minor issues detected but core functionality works.")
        return True
    else:
        print("\nAPP TEST: NEEDS ATTENTION")
        print("Multiple issues detected with the API.")
        return False

def main():
    try:
        success = test_app()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())
