# Personal LLM - Setup Embedded Python for Desktop Distribution
# Run this script BEFORE building the EXE: .\setup-python-embed.ps1
# Creates a self-contained Python in ui/python-embed/ that electron-builder bundles.

$ErrorActionPreference = "Stop"
$PYTHON_VERSION = "3.11.9"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-embed-amd64.zip"
$GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
$EMBED_DIR = Join-Path $PSScriptRoot "python-embed"
$ZIP_PATH = Join-Path $PSScriptRoot "python-embed.zip"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Personal LLM - Embedded Python Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = Join-Path $EMBED_DIR "python.exe"

# Step 1: Download Python embeddable
if (-Not (Test-Path $pythonExe)) {
    Write-Host "[1/4] Downloading Python $PYTHON_VERSION embeddable..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PYTHON_URL -OutFile $ZIP_PATH -UseBasicParsing

    Write-Host "[1/4] Extracting..." -ForegroundColor Yellow
    if (Test-Path $EMBED_DIR) { Remove-Item $EMBED_DIR -Recurse -Force }
    Expand-Archive -Path $ZIP_PATH -DestinationPath $EMBED_DIR -Force
    Remove-Item $ZIP_PATH -Force

    $pthFile = Get-ChildItem $EMBED_DIR -Filter "python*._pth" | Select-Object -First 1
    if ($pthFile) {
        $content = Get-Content $pthFile.FullName
        $content = $content -replace "^#import site", "import site"
        $content += "`nLib\site-packages"
        Set-Content $pthFile.FullName $content
        Write-Host "   Enabled site-packages in $($pthFile.Name)" -ForegroundColor Green
    }
} else {
    Write-Host "[1/4] Python embeddable already exists, skipping download." -ForegroundColor Green
}

# Step 2: Install pip
$pipExe = Join-Path $EMBED_DIR "Scripts\pip.exe"
if (-Not (Test-Path $pipExe)) {
    Write-Host "[2/4] Installing pip..." -ForegroundColor Yellow
    $getPipPath = Join-Path $EMBED_DIR "get-pip.py"
    Invoke-WebRequest -Uri $GET_PIP_URL -OutFile $getPipPath -UseBasicParsing
    & $pythonExe $getPipPath --no-warn-script-location 2>$null
    Remove-Item $getPipPath -Force -ErrorAction SilentlyContinue
    Write-Host "   pip installed." -ForegroundColor Green
} else {
    Write-Host "[2/4] pip already installed." -ForegroundColor Green
}

# Step 3: Install packages
Write-Host "[3/4] Installing Python packages..." -ForegroundColor Yellow

# Install everything EXCEPT llama-cpp-python first (these are pure Python or have wheels)
$simplePkgs = @(
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "huggingface-hub",
    "httpx",
    "requests",
    "PyPDF2",
    "starlette"
)
Write-Host "   Installing core packages..." -ForegroundColor Gray
& $pythonExe -m pip install $simplePkgs --no-warn-script-location --quiet 2>$null
Write-Host "   Core packages installed." -ForegroundColor Green

# Install llama-cpp-python from pre-built wheel (no compiler needed)
Write-Host "   Installing llama-cpp-python (pre-built wheel)..." -ForegroundColor Gray
try {
    & $pythonExe -m pip install llama-cpp-python --prefer-binary --no-warn-script-location --quiet 2>$null
    Write-Host "   llama-cpp-python installed from wheel." -ForegroundColor Green
} catch {
    Write-Host "   WARNING: llama-cpp-python failed. Trying alternative source..." -ForegroundColor Yellow
    try {
        & $pythonExe -m pip install llama-cpp-python --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cpu" --no-warn-script-location --quiet 2>$null
        Write-Host "   llama-cpp-python installed from alternative source." -ForegroundColor Green
    } catch {
        Write-Host "   WARNING: Could not install llama-cpp-python. Users will need to install it manually." -ForegroundColor Red
    }
}

# Step 4: Verify
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow
$version = & $pythonExe --version 2>&1
Write-Host "   $version" -ForegroundColor Green

try {
    $testFA = & $pythonExe -c "import fastapi; print('fastapi OK')" 2>&1
    Write-Host "   $testFA" -ForegroundColor Green
} catch {
    Write-Host "   fastapi: FAILED" -ForegroundColor Red
}

try {
    $testLC = & $pythonExe -c "import llama_cpp; print('llama_cpp OK')" 2>&1
    Write-Host "   $testLC" -ForegroundColor Green
} catch {
    Write-Host "   llama_cpp: FAILED (will be installed on first run)" -ForegroundColor Yellow
}

$totalSize = (Get-ChildItem $EMBED_DIR -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Done! Embedded Python: $([math]::Round($totalSize, 0)) MB" -ForegroundColor Cyan
Write-Host " Location: $EMBED_DIR" -ForegroundColor Cyan
Write-Host " Now run: npm run electron:build" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
