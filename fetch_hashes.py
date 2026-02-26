import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

models = [
    ('bartowski/Phi-3.1-mini-4k-instruct-GGUF', 'Phi-3.1-mini-4k-instruct-Q4_K_M.gguf'),
    ('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf'),
    ('bartowski/Mistral-7B-Instruct-v0.3-GGUF', 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'),
    ('bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF', 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf'),
    ('TheBloke/CodeLlama-7B-Instruct-GGUF', 'codellama-7b-instruct.Q4_K_M.gguf')
]

for repo, filename in models:
    url = f'https://huggingface.co/api/models/{repo}/tree/main'
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode())
            for item in data:
                if item.get('path') == filename:
                    sha256 = item.get("lfs", {}).get("oid")
                    size = item.get("lfs", {}).get("size")
                    print(f'{filename}: sha256="{sha256}", size={size}')
    except Exception as e:
        print(f"Error fetching {filename}: {e}")
