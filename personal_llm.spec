# -*- mode: python ; coding: utf-8 -*-
"""
Personal LLM — PyInstaller Build Spec (onedir mode)

Produces a folder with PersonalLLM.exe and all supporting files.
Launches instantly (no extraction step like onefile).

Usage:
    pyinstaller personal_llm.spec --clean --noconfirm

Note: Models (.gguf files) are NOT bundled. Place them in
      'personal_llm_models/' folder next to the built .exe.
"""

import sys
import os
import glob
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ─── Collect Dependencies ──────────────────────────────────────

datas = []
binaries = []
hiddenimports = []

# Packages that need full collection (data + binaries + submodules)
full_collect_packages = [
    'gradio',
    'gradio_client',
    'chromadb',
    'huggingface_hub',
    'uvicorn',
    'fastapi',
    'starlette',
    'webview',
    'sentence_transformers',
    'tokenizers',
    'pydantic',
    'anyio',
    'safehttpx',
    'groovy',
]

for package in full_collect_packages:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect {package}: {e}")

# ─── llama-cpp-python: Manually find DLLs ─────────────────────
# llama_cpp is a single .pyd file, not a package, so collect_all fails.
# We must manually locate it and any CUDA DLLs.

import importlib.util

# Find llama_cpp module
llama_spec = importlib.util.find_spec('llama_cpp')
if llama_spec and llama_spec.origin:
    llama_dir = os.path.dirname(llama_spec.origin)
    print(f"[SPEC] Found llama_cpp at: {llama_dir}")
    
    # Collect .dll and .pyd files from llama_cpp directory
    for ext in ('*.dll', '*.pyd', '*.so'):
        for f in glob.glob(os.path.join(llama_dir, '**', ext), recursive=True):
            binaries.append((f, os.path.join('llama_cpp', os.path.relpath(os.path.dirname(f), llama_dir))))
            print(f"[SPEC] Including binary: {f}")
    
    # Also look in a 'lib' subfolder (common for CUDA builds)
    lib_dir = os.path.join(llama_dir, 'lib')
    if os.path.exists(lib_dir):
        for f in glob.glob(os.path.join(lib_dir, '*')):
            binaries.append((f, 'llama_cpp/lib'))
            print(f"[SPEC] Including lib binary: {f}")

    hiddenimports.append('llama_cpp')
    hiddenimports.append('llama_cpp.llama')
    hiddenimports.append('llama_cpp.llama_cpp')
else:
    print("[SPEC] WARNING: llama_cpp not found! LLM inference will not work.")

# ─── CUDA DLLs (if NVIDIA GPU build) ──────────────────────────
# Search common locations for CUDA runtime DLLs
cuda_dll_patterns = [
    'cublas64_*.dll',
    'cublasLt64_*.dll', 
    'cudart64_*.dll',
    'cusparse64_*.dll',
]

cuda_search_paths = [
    os.path.join(os.environ.get('CUDA_PATH', ''), 'bin'),
    os.path.join(sys.prefix, 'Library', 'bin'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia'),
]

for search_path in cuda_search_paths:
    if os.path.exists(search_path):
        for pattern in cuda_dll_patterns:
            for f in glob.glob(os.path.join(search_path, '**', pattern), recursive=True):
                binaries.append((f, '.'))
                print(f"[SPEC] Including CUDA DLL: {f}")

# ─── Exclude PyTorch CUDA (keep CPU-only for embeddings) ──────
# PyTorch is only used by sentence-transformers for embeddings.
# We exclude the massive CUDA version to keep the build small.
excludes = [
    'torch.cuda',
    'torch.distributed',
    'torch.testing',
    'torchvision',
    'torchaudio', 
    'personal_llm_models',  # Models are external
]

# ─── Analysis ─────────────────────────────────────────────────

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'personal_llm',
        'personal_llm.config',
        'personal_llm.web_ui',
        'personal_llm.llm_engine',
        'personal_llm.chat_engine',
        'personal_llm.model_manager',
        'personal_llm.knowledge_base',
        'personal_llm.hardware',
        'PyPDF2',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PersonalLLM',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for debugging; set to False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here later, e.g. icon='assets/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PersonalLLM',
)
