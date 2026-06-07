# INSK RAG 데모 실행 (발표 시연용)
# 사용법: PowerShell에서  .\app\run_demo.ps1
#
# 키는 두 가지 방법 중 하나:
#   1) 이 스크립트에서 물어보면 붙여넣기 (아래 프롬프트)
#   2) 앱 켜진 뒤 왼쪽 사이드바 "OpenAI API Key" 칸에 붙여넣기
# 둘 다 안 넣으면 검색·리랭킹까지만 시연됨 (답변 생성 생략).

$ErrorActionPreference = "Stop"

# 스크립트 위치 기준 레포 루트로 이동
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# OpenAI 키: 이미 환경변수에 있으면 그대로, 없으면 물어봄(엔터로 건너뛰기 가능)
if (-not $env:OPENAI_API_KEY) {
    $key = Read-Host "OpenAI API Key 붙여넣기 (없으면 그냥 Enter -> 검색까지만)"
    if ($key) { $env:OPENAI_API_KEY = $key.Trim() }
}

if ($env:OPENAI_API_KEY) {
    Write-Host "[OK] OpenAI 키 설정됨 -> 답변 생성까지 작동" -ForegroundColor Green
} else {
    Write-Host "[INFO] 키 없음 -> 검색·리랭킹까지만. 앱 사이드바에서 나중에 넣어도 됨" -ForegroundColor Yellow
}

Write-Host "Streamlit 시작... 브라우저에서 http://localhost:8501 열림 (첫 로딩 ~30초)" -ForegroundColor Cyan
streamlit run app/streamlit_demo.py --server.port 8501
