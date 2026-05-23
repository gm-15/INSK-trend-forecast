# INSK Trend Forecast

> **한국어 AI 뉴스 도메인 Temporal-aware Hybrid Retrieval 기반 RAG 성능 최적화 연구**
> 부제: Embedding 모델·Cross-encoder Reranker·시간 가중치의 정량 ablation + Failure Case Analysis

상명대학교 시계열 데이터 수업 팀프로젝트 (2026 1학기).
[INSK](https://github.com/gm-15/INSK) 시스템에서 export한 한국어 AI 산업 뉴스 코퍼스를 사용해, 한국어 도메인에 최적화된 RAG retrieval 품질을 정량·정성 양쪽으로 측정한다.

---

## 🎯 한 줄 정의

뉴스 RAG에서 흔히 발생하는 3가지 실패(semantic ambiguity / stale retrieval / keyword mismatch)를 **Hybrid Retrieval + Cross-encoder Reranker + Temporal Weighting** 조합으로 개선하고, **Failure Case Analysis**로 개선 메커니즘을 입증.

---

## 📊 현재 데이터 (2026-05-23 기준)

| 항목 | 값 |
|---|:---:|
| 총 article | 약 290건 (매일 +30-50건 누적 중) |
| 시계열 범위 | 2026-05-19 ~ 진행 중 (5/19 이전 sparse) |
| 매체 | Naver 60% / AITimes 18% / TheGuru 18% (구 GoogleNewsClient 삭제됨) |
| 카테고리 | LLM / INFRA / AI Business / Telco (4종, v4 taxonomy 재설계됨) |
| 임베딩 | OpenAI `text-embedding-3-small` (1536d) |
| LLM 분석 | gpt-4o-mini (요약·인사이트·카테고리·태그) |
| **QA Benchmark** | **27개** (Strict 7 / Trend 14 / Negative 6) — `data/human_qa_benchmark_v1.jsonl` |

목표 누적 (W6, 6/25): **약 1,500-2,000건**

---

## 📁 폴더 구조

```
INSK-trend-forecast/
├── README.md                     ← 이 파일
├── .gitignore
├── data/
│   ├── human_qa_benchmark_v1.jsonl   ← ⭐ QA 평가셋 (Strict / Trend / Negative)
│   ├── human_qa_benchmark_v1.txt     ← 사람 읽기용 양식
│   ├── insk_corpus.parquet           ← (예정) INSK MySQL export
│   ├── article_embeddings.parquet    ← (예정) OpenAI 1536d 임베딩
│   └── article_analyses.parquet      ← (예정) summary·insight·category·tags
├── docs/
│   └── data_dictionary.md            ← 데이터 사전 + 알려진 한계
├── notebooks/
│   ├── 01_eda.ipynb                  ← (팀원 A 시작) EDA
│   ├── 02_retrieval_baseline.ipynb   ← (팀원 A) BM25 / Embedding / Hybrid
│   ├── 03_embedding_comparison.ipynb ← (팀원 A) OpenAI vs BGE-M3 vs ko-sroberta
│   ├── 04_hard_negative_mining.ipynb ← (팀원 B) Hard negative mining
│   ├── 05_reranker_finetune.ipynb    ← (팀원 B) Cross-encoder fine-tune
│   ├── 06_temporal_weighting.ipynb   ← (팀원 C) α·β·γ tuning
│   ├── 07_rag_generation.ipynb       ← (팀원 C) LLM 답변 생성
│   ├── 08_ragas_eval.ipynb           ← (팀원 C) RAGAS 4지표
│   └── 09_failure_analysis.ipynb     ← (팀원 C) 발표 핵심 슬라이드 자료
├── src/
│   ├── retrieval/                    ← retrieval 모듈
│   ├── reranker/                     ← reranker 모듈
│   ├── eval/                         ← 평가 유틸
│   └── rag_agent/                    ← RAG 통합
└── app/
    └── streamlit_demo.py             ← 최종 데모 (PM 통합)
```

---

## ⚡ 빠른 시작 (팀원 A·B·C)

### 1. 환경 셋업

```bash
# Python 3.10+ 권장
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# 핵심 라이브러리
pip install pandas numpy scikit-learn matplotlib jupyter
pip install sentence-transformers faiss-cpu rank-bm25
pip install openai ragas torch transformers
```

### 2. 데이터 로드

```python
import pandas as pd
import json

# QA benchmark (지금 사용 가능)
with open("data/human_qa_benchmark_v1.jsonl", encoding="utf-8") as f:
    qa = [json.loads(line) for line in f]

print(f"Total: {len(qa)} questions")
print(f"Strict: {sum(1 for q in qa if q['type']=='Strict')}")
print(f"Trend:  {sum(1 for q in qa if q['type']=='Trend')}")
print(f"Negative: {sum(1 for q in qa if q['type']=='Negative')}")
```

### 3. Article corpus 로드 (5/24 일요일 박건우 export 후)

```python
articles = pd.read_parquet("data/insk_corpus.parquet")
embeddings = pd.read_parquet("data/article_embeddings.parquet")
analyses = pd.read_parquet("data/article_analyses.parquet")
```

---

## 👥 팀 역할 분담 (4인)

| 역할 | 담당 | 핵심 산출물 |
|---|---|---|
| **박건우 (PM + Data)** | INSK export, QA benchmark, 통합 데모, 발표 | `data/`, `app/streamlit_demo.py`, 발표 슬라이드 |
| **팀원 A (Retrieval)** | BM25 / Embedding 3종 / Hybrid RRF + query 특성별 분석 | `notebooks/01-03` |
| **팀원 B (Reranker DL)** | Hard negative mining + Cross-encoder fine-tune + Improvement Ceiling 분석 | `notebooks/04-05`, `models/reranker_ft.pt` |
| **팀원 C (RAG + Eval)** | Temporal weighting + RAGAS + **Failure Case Analysis (발표 핵심)** | `notebooks/06-09` |

---

## 📐 평가 지표 (5가지)

QA benchmark의 3가지 type에 따라:

| 지표 | 적용 대상 | 의미 |
|---|:---:|---|
| **Recall@k** | Strict + Trend | Top-k 안에 gold article이 있나? |
| **MRR / nDCG** | Strict + Trend | 순위 품질 |
| **Faithfulness** | Strict + Trend (with ground_truth) | LLM 답변이 근거와 일치하나? |
| **Hallucination rate** | Negative | 답 없는 질문에 거짓말 하나? |
| **Abstention precision** | Negative | "모른다"고 답하는 능력 |

---

## 🗓️ 일정 (W1 ~ W6)

| 주차 | 핵심 마일스톤 |
|:---:|---|
| W1 (5/14-5/20) | ✅ 주제 확정, v4 인프라, QA benchmark v1 |
| **W2 (5/21-5/27)** | 🔴 **5/24 첫 데이터 export, 팀원 A·B·C 발진** |
| W3 (5/28-6/3) | Embedding 3종 비교, Reranker 학습, RAG baseline |
| W4 (6/4-6/10) | Temporal weighting, RAGAS 측정, B급 QA 검토 |
| W5 (6/11-6/17) | Failure Case Analysis, Ablation 표, 데모 |
| W6 (6/18-6/25) | 최종 발표 + 리포트 |

---

## ⚠️ 알려진 데이터 한계 (필수 인지)

상세는 [docs/data_dictionary.md](docs/data_dictionary.md) 참조. 요약:

1. **분류 편향**: gpt-4o-mini가 LLM 기사를 AI Business로 분류하는 경향 일부 남음 (v4 taxonomy 재설계로 완화 중)
2. **시계열 lumpy**: 5/19 이전은 sparse, 5/19 이후 연속 분포. **Temporal weighting 실험은 5/19 이후 데이터로**
3. **본문 truncate**: 6000자 이상 본문은 잘림 (임베딩은 정상)
4. **매체 편향**: Naver 60% (search API 풍부) / AITimes·TheGuru 각 18%

---

## 🎤 발표 narrative — Q&A 대응

| 예상 질문 | 답변 narrative |
|---|---|
| "QA 27개가 적지 않냐?" | "초기 50개 seed에서 corpus 실재성 검증 거쳐 A급 27개만 채택. '좋은 27개' > '애매한 50개'." |
| "왜 Strict/Trend/Negative 3분류?" | "현업 검색엔진(Google·Naver)도 query log와 benchmark를 분리. 평가는 엄격해야 retrieval 품질 비교 가능." |
| "단순 RAG와 뭐가 다른가?" | "3분류 + ground truth 문장 + Negative QA로 **hallucination·abstention까지 분리 측정**." |

---

## 🔗 참조

- INSK 본체 (백엔드): https://github.com/gm-15/INSK
- 멘토 피드백 changelog: [INSK/MENTOR_FEEDBACK_CHANGELOG.md](https://github.com/gm-15/INSK/blob/main/MENTOR_FEEDBACK_CHANGELOG.md)
- 평가 안내문: 상명대 시계열 데이터 수업 (2026-1)

---

## 📞 운영

- 매주 일요일 21시: 박건우가 `data/` 갱신 + 카톡 공지
- 회의: 매주 일요일 22시 (zoom 또는 카톡)
- GitHub Issue로 작업 이슈 관리
- 코드 commit은 자유롭게 push (main branch 직접 OK, 본인 작업 branch도 OK)
