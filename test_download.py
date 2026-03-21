import requests
import sys

def check_model(repo_id, filename):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"Checking URL: {url}")
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        if response.status_code == 200:
            print(f"✅ URL is valid. Content-Length: {response.headers.get('Content-Length')}")
        else:
            print(f"❌ URL returned {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test Qwen2.5-Coder-3B (Bartowski) - Should be public
    check_model("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF", "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf")
