# 팀원 A 작업 요약 — Retriever + Embedding 비교 실험

> 작성일: 2026-05-29  
> 담당: 팀원 A (ssm0521)  
> 역할: Retrieval 모델 구현 · Embedding 비교 실험 · Query 특성별 분석

---

## 1. 작업 개요

팀원 A가 담당한 파트는 **"어떤 검색 모델이 어떤 종류의 질문에 강한가?"** 를 실험으로 밝히는 것입니다.  
단순히 "누가 1등이냐"가 아닌, **query의 특성(어휘형/의미형, 한국어 고유명사 포함 여부, 영문 기술용어 포함 여부)** 에 따라 각 모델의 강점이 어떻게 달라지는지를 분석했습니다.

---

## 2. 사용한 데이터

| 파일 | 내용 | 비고 |
|------|------|------|
| `data/insk_corpus.parquet` | 뉴스 기사 406개 (article_id, title, source 등) | body 컬럼 없음 |
| `data/article_analyses.parquet` | 기사별 요약(5줄), 인사이트(1줄), 카테고리, 태그 | corpus와 article_id로 merge |
| `data/article_embeddings.parquet` | OpenAI 임베딩 406개 (1536차원) | article 166·167·168은 null |
| `data/human_qa_benchmark_v1.jsonl` | QA 27개 (질문, gold_articles, 카테고리, 타입) | Negative 6개는 gold 없음 |

### 검색용 텍스트 구성 방식
각 기사에서 아래 필드를 이어붙여 검색 대상 텍스트를 만들었습니다:
```
text = title + " " + summary + " " + insight + " " + tags
```

### QA 구조
| 카테고리 | QA ID | QA 수 | 타입 분포 |
|---------|-------|-------|---------|
| LLM | L1~L6 | 6개 | Strict 2, Trend 4 |
| INFRA | I1~I6 | 6개 | Strict 1, Trend 2, Negative 3 |
| AI Business | B1~B9 | 9개 | Strict 3, Trend 6 |
| Telco | T1~T6 | 6개 | Strict 1, Trend 2, Negative 3 |

> **Negative QA**: gold_articles=[] (corpus에 관련 기사 없음) → 지표 계산 제외, 별도 표기

---

## 3. 구현한 검색 모델 (5가지)

### 3-1. BM25 (키워드 기반)
- 라이브러리: `rank_bm25`
- 방식: 한글/영문 토큰화 → BM25Okapi 스코어링
- 특징: 정확한 단어 일치에 강함. 의미적 유사성은 파악 못함

### 3-2. OpenAI Embedding (text-embedding-3-small)
- 차원: 1536d
- corpus 임베딩: `article_embeddings.parquet`에 사전 저장된 값 사용 (API 호출 없음)
- 쿼리 임베딩: API 호출 (실험 시 27회 호출, 비용 ≈ $0.00001)
- 유사도: 코사인 유사도 (FAISS IndexFlatIP)

### 3-3. BGE-M3 (BAAI/bge-m3)
- 차원: 1024d
- 다국어 지원 모델 (한·영 모두 커버)
- corpus + 쿼리 모두 로컬 인코딩 (CPU)
- 특징: 영문 기술용어에 강함

### 3-4. ko-sroberta (jhgan/ko-sroberta-multitask)
- 차원: 768d
- 한국어 특화 Sentence-BERT 계열
- corpus + 쿼리 모두 로컬 인코딩 (CPU)
- 특징: 한국어 고유명사·회사명에 강함

### 3-5. Hybrid RRF (BM25 + BGE-M3)
- 방식: BM25 결과 50개 + BGE-M3 결과 50개를 Reciprocal Rank Fusion으로 합산
- RRF 공식: `score = Σ 1/(k + rank)`, k=60
- 특징: 어휘형 + 의미형 쿼리 모두 커버

---

## 4. 평가 지표

| 지표 | 설명 |
|------|------|
| **Recall@5** | 상위 5개 결과 내 gold 기사 포함 비율 (주 지표) |
| MRR | 첫 번째 정답이 나온 순위의 역수 |
| nDCG@5 | 순위 가중 정확도 |

---

## 5. 전체 실험 결과

### 5-1. 전체 평균 Recall@5 (Positive QA 21개 기준)

| 모델 | Recall@5 | 순위 |
|------|----------|------|
| **Hybrid (BM25+BGE-M3)** | **0.417** | 🥇 |
| BGE-M3 | 0.401 | 🥈 |
| ko-sroberta | 0.353 | 🥉 |
| BM25 | 0.306 | 4위 |
| OpenAI (text-embedding-3-small) | 0.206 | 5위 |

> OpenAI 범용 임베딩이 **가장 낮음** → 한국어 AI 뉴스 도메인에서 도메인 특화 모델(BGE-M3, ko-sroberta)이 우세함

---

## 6. Query 특성별 분석 결과 (팀 A 핵심 기여)

### 6-1. Query 스타일: Lexical vs Semantic

| Query 스타일 | 쿼리 수 | BM25 | OpenAI | BGE-M3 | ko-sroberta | Hybrid | **최적** |
|------------|--------|------|--------|--------|-------------|--------|---------|
| **Lexical** (특정 엔티티/키워드 중심) | 10개 | 0.408 | 0.233 | 0.442 | 0.508 | **0.575** | **Hybrid** |
| **Semantic** (트렌드·동향·개념 중심) | 11개 | 0.212 | 0.182 | **0.364** | 0.212 | 0.273 | **BGE-M3** |

- **Lexical 쿼리** 예시: "OpenAI가 최근 발표한 신모델은?", "NVIDIA AI 칩 동향은?", "SKT AI 서비스·전략은?"
- **Semantic 쿼리** 예시: "AI 에이전트 시장 최근 동향은?", "글로벌 AI 정책·규제 동향은?", "빅테크 AI 투자 규모는?"

### 6-2. 언어 특성: 한국어 고유명사 vs 영문 기술용어

| 언어 특성 | 쿼리 수 | BM25 | OpenAI | BGE-M3 | ko-sroberta | Hybrid | **최적** |
|---------|--------|------|--------|--------|-------------|--------|---------|
| 🇰🇷 **한국어 고유명사 포함** | 4개 | 0.062 | 0.125 | 0.521 | **0.688** | 0.562 | **ko-sroberta** |
| 🔤 **영문 기술용어 포함** | 9개 | 0.343 | 0.204 | **0.528** | 0.361 | 0.343 | **BGE-M3** |
| 💬 **언어 특성 없음 (일반 의미론)** | 10개 | 0.333 | 0.250 | 0.267 | 0.217 | **0.367** | **Hybrid** |

- 한국어 고유명사 예시: 삼성전자, SKT, KT, LG U+, 한국 정부
- 영문 기술용어 예시: OpenAI, Anthropic, Gemini, NVIDIA, HBM, M&A, Edge AI

### 6-3. QA 타입별

| QA 타입 | 쿼리 수 | BM25 | OpenAI | BGE-M3 | ko-sroberta | Hybrid | **최적** |
|--------|--------|------|--------|--------|-------------|--------|---------|
| **Strict** (단일 정답 중심) | 7개 | 0.214 | 0.143 | **0.571** | 0.500 | 0.500 | **BGE-M3** |
| **Trend** (복합 트렌드 파악) | 14개 | 0.351 | 0.238 | 0.315 | 0.280 | **0.375** | **Hybrid** |
| Negative (gold 없음) | 6개 | — | — | — | — | — | 평가 제외 |

### 6-4. Category별

| Category | 쿼리 수 | BM25 | OpenAI | BGE-M3 | ko-sroberta | Hybrid | **최적** |
|---------|--------|------|--------|--------|-------------|--------|---------|
| LLM | 6개 | 0.292 | 0.222 | **0.347** | 0.208 | 0.236 | **BGE-M3** |
| INFRA | 3개 | 0.222 | 0.111 | **0.667** | **0.667** | **0.667** | BGE/KSR/HYB 공동 |
| AI Business | 9개 | 0.296 | 0.204 | 0.370 | 0.333 | **0.407** | **Hybrid** |
| Telco | 3개 | 0.444 | 0.278 | 0.333 | 0.389 | **0.556** | **Hybrid** |

### 6-5. 개별 QA 전체 결과표

| QID | 질문 요약 | 스타일 | KOR | ENG | BM25 | OpenAI | BGE-M3 | ko-sroberta | Hybrid | Best |
|-----|---------|-------|-----|-----|------|--------|--------|-------------|--------|------|
| L1 | OpenAI 신모델·기능 | lexical | N | Y | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| L2 | AI 에이전트 시장 동향 | semantic | N | N | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| L3 | Anthropic 매출·사업 | lexical | N | Y | 0.50 | 0.50 | **1.00** | 0.50 | 0.50 | BGE-M3 |
| L4 | Gemini·구글 AI 전략 | lexical | Y | Y | 0.25 | 0.50 | **0.75** | **0.75** | 0.25 | BGE/KSR |
| L5 | 멀티모달 AI 사례 | semantic | N | N | 0.33 | 0.33 | 0.33 | 0.00 | 0.33 | BM25 |
| L6 | 오픈소스 LLM 동향 | lexical | N | Y | **0.67** | 0.00 | 0.00 | 0.00 | 0.33 | BM25 |
| I1 | 삼성전자 AI 반도체 | lexical | Y | N | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | BGE/KSR/HYB |
| I2 | SK하이닉스·HBM | lexical | Y | Y | — | — | — | — | — | Negative |
| I3 | NVIDIA AI 칩 동향 | lexical | N | Y | **0.67** | 0.33 | 0.33 | 0.33 | **0.67** | BM25/HYB |
| I4 | 국내 AI 반도체 스타트업 | semantic | N | N | — | — | — | — | — | Negative |
| I5 | AI 반도체 미·중 경쟁 | semantic | N | N | — | — | — | — | — | Negative |
| I6 | on-device·Edge AI | semantic | N | Y | 0.00 | 0.00 | **0.67** | **0.67** | 0.33 | BGE/KSR |
| B1 | OpenAI IPO·매출 | lexical | N | Y | **1.00** | 0.50 | **1.00** | **1.00** | **1.00** | 전 모델 우수 |
| B2 | 최근 AI M&A 사례 | semantic | N | Y | 0.00 | 0.00 | **0.67** | 0.00 | 0.00 | BGE-M3 |
| B3 | AI 스타트업 투자 | semantic | N | N | 0.00 | **0.33** | 0.00 | 0.00 | 0.00 | OpenAI |
| B4 | 글로벌 AI 정책·규제 | semantic | N | N | **0.67** | 0.33 | 0.33 | **0.67** | 0.33 | BM25/KSR |
| B5 | 한국 정부 AI 정책 | lexical | Y | N | 0.00 | 0.00 | 0.00 | **1.00** | **1.00** | KSR/HYB |
| B6 | 생성형 AI 시장 동향 | semantic | N | N | 0.33 | 0.00 | 0.00 | 0.00 | 0.33 | BM25/HYB |
| B7 | 글로벌 AI 허브 경쟁 | semantic | N | N | 0.33 | **0.67** | **0.67** | 0.33 | **0.67** | OpenAI/BGE/HYB |
| B8 | 빅테크 AI 투자 규모 | semantic | N | N | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | — |
| B9 | AI 일자리·고용 영향 | semantic | N | N | 0.33 | 0.00 | **0.67** | 0.00 | 0.33 | BGE-M3 |
| T1 | SKT AI 서비스·전략 | lexical | Y | Y | 0.00 | 0.00 | **0.33** | 0.00 | 0.00 | BGE-M3 |
| T2 | KT AI 사업 동향 | lexical | Y | N | — | — | — | — | — | Negative |
| T3 | LG U+ AI 서비스 | lexical | Y | Y | — | — | — | — | — | Negative |
| T4 | 5G·6G와 AI 결합 | semantic | N | Y | — | — | — | — | — | Negative |
| T5 | 통신사 데이터센터 | lexical | N | N | **1.00** | 0.50 | 0.00 | 0.50 | **1.00** | BM25/HYB |
| T6 | 통신사 AI 에이전트 | semantic | N | N | 0.33 | 0.33 | **0.67** | **0.67** | **0.67** | BGE/KSR/HYB |

---

## 7. 핵심 인사이트 요약

```
1. 전체 최고: Hybrid (RRF) — Recall@5 = 0.417

2. Query 스타일별
   - Lexical (특정 엔티티 포함) → Hybrid 추천  (0.575)
   - Semantic (트렌드/동향 질문) → BGE-M3 추천 (0.364)

3. 언어 특성별
   - 한국어 고유명사 (삼성전자, SKT 등) → ko-sroberta 압도적 (0.688)
   - 영문 기술용어 (OpenAI, NVIDIA 등)  → BGE-M3 최강       (0.528)
   - 일반 의미론 (특정 기업명 없는 트렌드) → Hybrid 추천    (0.367)

4. OpenAI text-embedding-3-small은 5위 (0.206)
   → 범용 임베딩은 한국어 AI 뉴스 도메인에서 도메인 특화 모델에 밀림
   → B3(AI 스타트업 투자) 1케이스에서만 단독 최고

5. BM25는 lexical 쿼리에서 여전히 경쟁력 있음
   → L6(오픈소스 LLM), I3(NVIDIA), T5(데이터센터)에서 단독 또는 공동 1위
```

---

## 8. 생성된 파일 목록

| 파일 | 용도 | 대상 |
|------|------|------|
| `notebooks/01_eda.ipynb` | 데이터 구조 탐색 (EDA) | 팀 전체 |
| `notebooks/02_retrieval_baseline.ipynb` | 5개 모델 baseline 실험 | 팀 전체 |
| `notebooks/03_query_characteristic_analysis.ipynb` | Query 특성별 분석 노트북 | 팀 전체 |
| `data/retrieval_top10_for_reranker.jsonl` | **팀 B용** Hard Negative Mining 입력 파일 | **Team B** |
| `scripts/run_characteristic_analysis.py` | 특성 분석 실험 스크립트 (재현용) | 팀 전체 |

### `retrieval_top10_for_reranker.jsonl` 구조 (Team B 참고)
```json
{
  "qid": "L3",
  "question": "Anthropic 매출·사업 동향은?",
  "gold_articles": [427, 345],
  "retrieved": {
    "BM25": [427, 291, ...],
    "BGE-M3": [427, 345, ...],
    "ko-sroberta": [...],
    "Hybrid": [...]
  }
}
```
각 모델별 상위 10개 retrieved article_id가 들어있습니다.

---

## 9. 환경 설정 (재현 방법)

```bash
pip install rank-bm25 sentence-transformers openai python-dotenv pandas pyarrow faiss-cpu
```

```
# .env 파일 (git 제외됨, .env.example 참고)
OPENAI_API_KEY=sk-proj-...
```

```bash
# 특성 분석 재실행
python scripts/run_characteristic_analysis.py

# 노트북 실행
jupyter notebook notebooks/03_query_characteristic_analysis.ipynb
```

---

## 10. 남은 작업

| 작업 | 상태 | 비고 |
|------|------|------|
| 03 노트북 시각화 (bar chart, heatmap) | 🔲 예정 | 발표 슬라이드용 |
| GitHub push | 🔲 예정 | collaborator 권한 확인 후 |
| Team B에 jsonl 파일 전달 | 🔲 예정 | GitHub push 후 자동 공유 |
| Team C에 retrieval 결과 전달 | 🔲 나중 | RAGAS 평가 때 필요 |

---

*작성: 팀원 A / 문의사항은 카카오톡으로*
