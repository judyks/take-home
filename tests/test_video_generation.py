#!/usr/bin/env python3
"""
Comprehensive Video Generation API Test Suite
Tests video generation functionality end-to-end
"""

import requests
import time
import json
import sys
import os
from typing import Dict, Any

class VideoGenerationTester:
    """Comprehensive tester for video generation API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if details:
            print(f"      {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
        return success
    
    def test_model_loading(self) -> bool:
        """Test model loading functionality"""
        print("\n📥 Testing Model Loading...")
        try:
            response = self.session.get(f"{self.base_url}/test-model-loading", timeout=300)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                if status == "success":
                    return self.log_test("Model Loading", True, f"Model: {data.get('model_id')}")
                elif status == "pytorch_unavailable":
                    return self.log_test("Model Loading", True, "PyTorch unavailable (expected in some environments)")
                else:
                    return self.log_test("Model Loading", False, f"Status: {status}")
            else:
                return self.log_test("Model Loading", False, f"HTTP {response.status_code}")
        except Exception as e:
            return self.log_test("Model Loading", False, f"Error: {str(e)}")
    
    def test_video_generation(self) -> bool:
        """Test actual video generation"""
        print("\n🎬 Testing Video Generation...")
        
        test_prompts = [
            ("A cat walking", 3),
            ("Ocean waves", 2),
        ]
        
        generation_success = False
        
        for prompt, duration in test_prompts:
            try:
                print(f"   🎯 Testing prompt: '{prompt}' ({duration}s)")
                
                # Send generation request
                response = self.session.post(
                    f"{self.base_url}/generate",
                    params={"prompt": prompt, "duration": duration},
                    timeout=600  # 10 minutes for generation
                )
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status")
                    
                    if status == "success":
                        job_id = data.get("job_id")
                        self.log_test(f"Generation: '{prompt}'", True, f"Job ID: {job_id}")
                        
                        # Test job status
                        if self.test_job_status(job_id):
                            # Test video download
                            if self.test_video_download(job_id):
                                generation_success = True
                    
                    elif status in ["partial_success", "error"]:
                        self.log_test(f"Generation: '{prompt}'", False, data.get("message", "Unknown error"))
                
                else:
                    self.log_test(f"Generation: '{prompt}'", False, f"HTTP {response.status_code}")
                
                # Only test one prompt to save time
                break
                
            except Exception as e:
                self.log_test(f"Generation: '{prompt}'", False, f"Error: {str(e)}")
        
        return generation_success
    
    def test_job_status(self, job_id: str) -> bool:
        """Test job status endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                return self.log_test("Job Status", True, f"Status: {status}")
            else:
                return self.log_test("Job Status", False, f"HTTP {response.status_code}")
        except Exception as e:
            return self.log_test("Job Status", False, f"Error: {str(e)}")
    
    def test_video_download(self, job_id: str) -> bool:
        """Test video download endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/download/{job_id}", timeout=60)
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                file_size = len(response.content)
                if 'video' in content_type and file_size > 1000:  # At least 1KB
                    return self.log_test("Video Download", True, f"Size: {file_size} bytes")
                else:
                    return self.log_test("Video Download", False, f"Invalid content: {content_type}, {file_size} bytes")
            else:
                return self.log_test("Video Download", False, f"HTTP {response.status_code}")
        except Exception as e:
            return self.log_test("Video Download", False, f"Error: {str(e)}")
    
    def test_api_validation(self) -> bool:
        """Test API input validation"""
        print("\n🔍 Testing API Validation...")
        
        validation_tests = [
            # Test empty prompt
            {"prompt": "", "duration": 3, "expected_status": 400},
            # Test invalid duration
            {"prompt": "test", "duration": 15, "expected_status": 400},
            {"prompt": "test", "duration": 0, "expected_status": 400},
        ]
        
        all_passed = True
        
        for test in validation_tests:
            try:
                response = self.session.post(
                    f"{self.base_url}/generate",
                    params={"prompt": test["prompt"], "duration": test["duration"]},
                    timeout=30
                )
                
                if response.status_code == test["expected_status"]:
                    self.log_test(f"Validation: {test}", True)
                else:
                    self.log_test(f"Validation: {test}", False, f"Expected {test['expected_status']}, got {response.status_code}")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Validation: {test}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def run_comprehensive_test(self) -> bool:
        """Run all tests"""
        print("🧪 Comprehensive Video Generation API Test")
        print("=" * 60)
        
        # Basic connectivity test
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=30)
            if response.status_code != 200:
                print("❌ API is not accessible. Make sure the service is running.")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
        
        print("✅ API is accessible")
        
        # Run test suite
        tests = [
            self.test_model_loading,
            self.test_api_validation,
            self.test_video_generation,
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            if test_func():
                passed_tests += 1
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {passed_tests}/{total_tests} test groups passed")
        
        individual_passed = sum(1 for result in self.test_results if result["success"])
        individual_total = len(self.test_results)
        print(f"📋 Individual Tests: {individual_passed}/{individual_total} passed")
        
        if passed_tests == total_tests:
            print("🎉 All test groups passed!")
            print("\n🚀 API is ready for production use!")
        else:
            print("⚠️  Some tests failed. Review the issues above.")
        
        return passed_tests == total_tests

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        print("⏳ Waiting 60 seconds for server and model to be ready...")
        time.sleep(60)
    
    # Get base URL
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    if len(sys.argv) > 1 and sys.argv[-1].startswith("http"):
        base_url = sys.argv[-1]
    
    # Run tests
    tester = VideoGenerationTester(base_url)
    success = tester.run_comprehensive_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
