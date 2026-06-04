# INSK RAG 데모 (서비스 시연)

중간발표 슬라이드 8 "실제 작동하는 서비스 시연" 충족용 Streamlit 앱.
INSK 운영 데이터로 팀 A·B·C 파이프라인을 라이브로 시연한다.

## 파이프라인

```
질문 입력
  → 1단계 Hybrid 검색 (BM25 + BGE-M3, RRF)      [팀 A]
  → 2단계 Reranker (klue/bert-base fine-tuned) + Temporal decay   [팀 B + C]
  → 3단계 GPT-4o-mini 답변 생성 (근거 인용 + 환각 방지)   [팀 C]
```

## 실행 방법

### 1. 의존성 설치
```bash
pip install -r app/requirements.txt
```

### 2. Reranker 모델 배치 (445MB, git 제외)
`reranker_ft.pt`를 아래 중 한 곳에 둔다 (앱이 자동 탐색):
- `src/reranker/reranker_ft.pt`  ← 권장
- 환경변수 `RERANKER_PATH`로 직접 지정
- `~/Downloads/src-.../src/reranker/reranker_ft.pt`

모델이 없으면 reranker 단계를 건너뛰고 검색 순서를 유지한다 (데모는 계속 동작).

### 3. OpenAI 키 설정 (3단계 답변 생성용)
```powershell
$env:OPENAI_API_KEY = "sk-..."
```
키가 없으면 1·2단계(검색·리랭킹)까지만 시연된다.

### 4. 실행
```bash
streamlit run app/streamlit_demo.py
```

## 사용법

- **질문 입력**: 아무 질문이나 직접 입력 가능 (예: "삼성전자 AI 반도체 전략은?")
- **벤치마크 질문**: 사이드바에서 27개 QA 중 선택 → 정답 article이 ⭐로 표시되어 검색 정확도 확인 가능
- **결과 화면**: 1차 검색 10개 → 리랭킹 후 top-3 (점수 포함) → 최종 답변 (근거 인용)

## 발표 시연 팁

1. 사이드바에서 벤치마크 질문 "삼성전자 AI 반도체 전략은?" 선택 (정답 잘 맞는 케이스)
2. 1차 검색 vs 리랭킹 후 ⭐(정답) 위치 변화를 보여주기 → "리랭커가 정답을 위로 올린다"
3. Failure 케이스로 "한국 정부 AI 정책은?" → 검색이 엉뚱한 뉴스 → AI가 "관련 내용 없음" 정직하게 답변 → "환각 대신 회피" 시연

## 주의

- 첫 실행 시 BGE-M3(~2GB) 다운로드 → 시간 소요 (이후 캐시)
- corpus 임베딩은 최초 1회 계산 후 캐시
- 현재 검색 대상: 데이터 사전 §4.2·4.6 필터 적용분 (5/19 이후 + Naver/최신 RSS)
