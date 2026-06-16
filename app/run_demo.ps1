# INSK RAG Demo launcher (ASCII only - avoids PowerShell 5.1 codepage issues)
# Usage:  .\app\run_demo.ps1
#
# OpenAI key (for answer generation) - two ways:
#   1) paste when this script prompts below
#   2) paste into the left sidebar "OpenAI API Key" box after the app opens
# If neither, the demo runs search + reranking only (no answer generation).

$ErrorActionPreference = "Stop"

# move to repo root (parent of this script's folder)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# OpenAI key: keep existing env var, else prompt (Enter to skip)
if (-not $env:OPENAI_API_KEY) {
    $key = Read-Host "Paste OpenAI API Key (or just Enter -> search only)"
    if ($key) { $env:OPENAI_API_KEY = $key.Trim() }
}

if ($env:OPENAI_API_KEY) {
    Write-Host "[OK] OpenAI key set -> answer generation enabled" -ForegroundColor Green
} else {
    Write-Host "[INFO] No key -> search + rerank only. You can paste it in the sidebar later." -ForegroundColor Yellow
}

Write-Host "Starting Streamlit... open http://localhost:8501 (first load ~30s)" -ForegroundColor Cyan
streamlit run app/streamlit_demo.py --server.port 8501
