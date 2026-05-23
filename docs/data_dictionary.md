# 데이터 사전 (Data Dictionary)

INSK 시스템에서 export한 한국어 AI 산업 뉴스 코퍼스의 스키마·도메인·한계.

---

## 1. 파일 목록

| 파일 | 설명 | 상태 |
|---|---|:---:|
| `data/human_qa_benchmark_v1.jsonl` | QA 평가셋 27개 (Strict/Trend/Negative) | ✅ 사용 가능 |
| `data/human_qa_benchmark_v1.txt` | 사람 가독성 양식 | ✅ |
| `data/insk_corpus.parquet` | articles 테이블 | ⏳ 5/24 export 예정 |
| `data/article_analyses.parquet` | LLM 분석 결과 | ⏳ 5/24 |
| `data/article_embeddings.parquet` | OpenAI 1536d 임베딩 | ⏳ 5/24 |
| `data/article_feedbacks.parquet` | 좋아요·싫어요 (현재 거의 비어있음) | ⏳ (선택) |
| `data/keywords.parquet` | 검색 키워드 23개 | ⏳ |

---

## 2. 스키마

### `articles.parquet`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| article_id | BIGINT (PK) | 고유 ID |
| title | VARCHAR(500) | 한국어 95%+ (영문 모델·회사명 일부 혼합) |
| body | TEXT | 본문 (6000자에서 truncate됨, §4.3 참조) |
| original_url | TEXT | 원본 기사 URL |
| source | VARCHAR(100) | "Naver" / "AITimes" / "TheGuru" |
| published_at | DATETIME | 매체 발행 시각 |
| created_at | TIMESTAMP | INSK 수집 시각 |
| country | VARCHAR(50) | 거의 "KR" |
| language | VARCHAR(50) | 거의 "ko" |

### `article_analyses.parquet`

| 컬럼 | 설명 |
|---|---|
| analysis_id (PK) | |
| article_id (FK) | |
| summary | gpt-4o-mini 한국어 5줄 요약 |
| insight | SKT 전략 관점 인사이트 (1줄) |
| **category** | **LLM / INFRA / AI Business / Telco** (4종, v4 taxonomy 재설계됨) |
| tags | JSON 배열 `["키워드1", "키워드2", "키워드3"]` |
| user_id | 분석한 사용자 (필터 시 무시 가능) |
| created_at | |

> ⚠️ 구 카테고리 `AI Ecosystem`은 모두 `AI Business`로 마이그레이션됨 (2026-05-22).

### `article_embeddings.parquet`

| 컬럼 | 설명 |
|---|---|
| embedding_id (PK) | |
| article_id (FK) | |
| embedding_json | **1536차원 float**, JSON 문자열 직렬화 |

**로드 예시**:
```python
import json
import numpy as np

df = pd.read_parquet("data/article_embeddings.parquet")
df["embedding"] = df["embedding_json"].apply(lambda s: np.array(json.loads(s)))
# df["embedding"].iloc[0].shape → (1536,)
```

### `keywords.parquet` (검색 키워드 23개)

| 카테고리 (의도) | 키워드 |
|---|---|
| LLM | LLM, Multimodal LLM, GPT-4, Claude AI, Gemini, RLHF, 오픈소스 LLM, LLM 파인튜닝, AI inference, vector DB |
| INFRA | AI 반도체, GPU 시장, Edge AI, AI semiconductor, on-device AI, AI datacenter |
| AI Business | 생성형 AI, AI 에이전트, OpenAI, AI agent, Multimodal AI |
| Telco | 5G 네트워크, 통신 AI |

---

## 3. 도메인 컨텍스트

### 4 카테고리 정의 (v4 taxonomy 재설계, 2026-05-22)

| 카테고리 | 의도 | 예시 |
|---|---|---|
| **LLM** | foundation model·LLM 기술 자체 | "GPT-5 학습 방법", "Claude 파인튜닝", "RLHF" |
| **INFRA** | AI 하드웨어·반도체·데이터센터 | "NVIDIA H200", "HBM 시장", "AI 데이터센터" |
| **AI Business** | 산업·정책·투자·시장 (비기술) | "OpenAI IPO", "정부 AI 정책", "스타트업 투자" |
| **Telco** | 통신사·네트워크 | "SKT AI 서비스", "5G", "통신 AI" |

### 10 부서 ENUM
T_CLOUD / T_NETWORK_INFRA / T_HR / T_AI_SERVICE / T_MARKETING / T_STRATEGY / T_ENTERPRISE_B2B / T_PLATFORM_DEV / T_TELCO_MNO / T_FINANCE

→ INSK 시스템의 부서 추천 기능용. 학교 프로젝트 retrieval 실험에는 주로 무시.

---

## 4. ⚠️ 알려진 한계 — 실험 영향

### 4.1 분류 편향 (mitigated, 일부 남음)

**현황**: v4 taxonomy 재설계로 일부 완화됨.
- 2026-05-22 마이그레이션 후 LLM 카테고리 4% → 20% 증가
- 그러나 신규 trigger에서 AI Business 비율 여전히 57-71%

**영향**:
- ❌ retrieval 실험 (BM25 / Embedding / Reranker)은 본문 기반 → 영향 없음
- ✅ category 컬럼 필터링은 LLM 부족할 수 있음 → 본문·키워드 매칭으로 우회 권장

### 4.2 시계열 lumpy

```
2025-12-23 : 50건
2025-12-25 : 35건
2026-05-19 ~ : 매일 누적 시작 (연속 분포)
```

→ **Temporal weighting 실험은 2026-05-19 이후 데이터로**. 그 이전은 노이즈.

### 4.3 본문 truncate 6000자

OpenAI 임베딩 호출 시 본문 6000자 이상은 잘림. 임베딩은 정상 동작.
- `articles.body`도 잘려 있을 가능성
- long-form 분석에는 부적합

### 4.4 매체 편향

| 매체 | 비율 | 비고 |
|---|:---:|---|
| Naver | 60% | search API 풍부 |
| AITimes | 18% | RSS 매일 1-2건 갱신 |
| TheGuru | 18% | 동일 |

→ retrieval 실험 시 매체별 균형 맞춤 평가 권장.

### 4.5 외부 매체 검색 한계

- Naver `display=10` + `sort=sim` + `start` 없음 → 키워드당 최대 10건 풀
- AITimes/TheGuru RSS는 매체 최신 발행분만
- **과거 데이터 백필 불가능** → 시계열 길이는 INSK 가동 시작 시점부터

---

## 5. ⭐ QA Benchmark (`human_qa_benchmark_v1.jsonl`)

27개 질문, 3분류:

| Type | 개수 | 평가 지표 |
|---|:---:|---|
| **Strict** (7) | 정답 1-2개 명확 | Recall@k, Precision, Reranker 차별화 |
| **Trend** (14) | 정답 다수 (synthesis) | Recall@k, Faithfulness |
| **Negative** (6) | 정답 없음 | Hallucination rate, Abstention precision |

### JSONL 스키마

```json
{
  "id": "L1",
  "question": "...",
  "category": "LLM",
  "type": "Strict" | "Trend" | "Negative",
  "gold_articles": [519, 317],
  "ground_truth": "정답 문장 1-2줄",
  "notes": "메모"
}
```

### 로드 예시

```python
import json
qa = [json.loads(l) for l in open("data/human_qa_benchmark_v1.jsonl", encoding="utf-8")]

# 타입별 분리
strict = [q for q in qa if q["type"] == "Strict"]   # 7개
trend = [q for q in qa if q["type"] == "Trend"]     # 14개
negative = [q for q in qa if q["type"] == "Negative"]  # 6개

# Negative QA — hallucination 평가용
for q in negative:
    print(q["question"], "→ 기대 답: '모른다'")
```

---

## 6. 데이터 갱신 일정

| 일자 | 작업 | 누적 예상 |
|:---:|---|:---:|
| 2026-05-24 (일) | 첫 export | 약 350건 |
| 5/31 (일) | 주간 갱신 | 약 600건 |
| 6/7 (일) | 주간 갱신 | 약 900건 |
| 6/14 (일) | 주간 갱신 | 약 1,200건 |
| 6/21 (일) | 최종 export | 약 1,500-2,000건 |

박건우가 매주 일요일 21시 이전 업로드 + 카톡 공지.

---

## 7. 발표 narrative (Q&A 대응)

| 질문 | 답변 |
|---|---|
| "QA 27개 적지 않냐?" | "초기 50개에서 corpus 실재성 검증, A급 27개만 채택. '좋은 27개' > '애매한 50개'" |
| "분류 편향은?" | "v4 재설계로 일부 완화. retrieval은 본문 기반이라 영향 없음. Failure analysis 자료로도 활용" |
| "시계열 짧지 않냐?" | "Temporal weighting은 보조 layer (1-2슬라이드). 메인은 retrieval quality 개선" |
| "왜 INSK 데이터만?" | "운영 시스템이라 시연 가능 + 분석·임베딩이 이미 완성 → 학기 첫 1-2주 데이터셋 작업 절약" |
