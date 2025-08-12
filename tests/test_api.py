import pytest
import os
import requests

@pytest.fixture
def base_url():
    return os.getenv("API_BASE_URL", "http://localhost:8000")

def test_health_endpoint(base_url):
    response = requests.get(f"{base_url}/health", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "gpu_available" in data
    print("test_health_endpoint passed")

def test_root_endpoint(base_url):
    response = requests.get(f"{base_url}/", timeout=10)
    assert response.status_code == 200
    print("test_root_endpoint passed")

def test_docs_endpoint(base_url):
    response = requests.get(f"{base_url}/docs", timeout=10)
    assert response.status_code == 200
    print("test_docs_endpoint passed")

def test_model_status_endpoint(base_url):
    response = requests.get(f"{base_url}/model-status", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    print("test_model_status_endpoint passed")

import pytest

@pytest.mark.parametrize("prompt", [
    "A cat stretching",
    "A green ball",
    "A tree in the wind"
])
def test_generate_success(base_url, prompt):
    print(f"Generating video.. (prompt: '{prompt}')")
    payload = {"prompt": prompt, "duration": 3}
    response = requests.post(f"{base_url}/generate", params=payload, timeout=30)
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "success"
    print(f"test_generate_success passed for prompt: '{prompt}'")
    job_id = data.get("job_id")
    if job_id:
        video_url = f"{base_url}/preview/{job_id}"
        print(f"\nVideo preview URL for '{prompt}': {video_url}\n")

@pytest.mark.parametrize("payload,expected_status", [
    ({"prompt": "", "duration": 3}, 400),
    ({"prompt": "test", "duration": 0}, 400),
    ({"prompt": "test", "duration": 1000}, 400),
])
def test_generate_invalid(base_url, payload, expected_status):
    response = requests.post(f"{base_url}/generate", params=payload, timeout=10)
    assert response.status_code == expected_status
    print(f"test_generate_invalid passed for payload: {payload}, expected_status: {expected_status}")
