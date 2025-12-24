# ================================
# PDA Agent Universal Launcher
# ================================

Write-Host "=== PDA Agent Launcher ===" -ForegroundColor Cyan

# 1. Ga naar de map waar dit script staat
Set-Location "$PSScriptRoot"

# 2. Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Please install Python from https://www.python.org" -ForegroundColor Red
    exit
}

# 3. Check Ollama
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. Install it from https://ollama.com" -ForegroundColor Red
    exit
}

# 4. Check of Ollama draait
try {
    Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null
} catch {
    Write-Host "Ollama is installed but not running. Start Ollama first." -ForegroundColor Red
    exit
}

# 5. Check Mistral model
$models = ollama list
if ($models -notmatch "mistral") {
    Write-Host "Mistral model not found. Run: ollama pull mistral" -ForegroundColor Yellow
    exit
}

# 6. Dependencies installeren indien nodig
if (Test-Path "requirements.txt") {
    Write-Host "Checking Python dependencies..."
    python -m pip install -r requirements.txt
}

# 7. Start de agent
Write-Host "Starting PDA Agent..." -ForegroundColor Green
python api.py