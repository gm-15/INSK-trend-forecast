# 데이터 사전 (Data Dictionary)

INSK 시스템에서 export한 한국어 AI 산업 뉴스 코퍼스의 스키마·도메인·한계.

---

## 1. 파일 목록

| 파일 | 설명 | 상태 (2026-05-27) |
|---|---|:---:|
| `data/human_qa_benchmark_v1.jsonl` | QA 평가셋 27개 (Strict/Trend/Negative) | ✅ |
| `data/human_qa_benchmark_v1.txt` | 사람 가독성 양식 | ✅ |
| `data/insk_corpus.parquet` | articles 테이블 (406건) | ✅ 5/27 export |
| `data/article_analyses.parquet` | LLM 분석 결과 (406건) | ✅ 5/27 |
| `data/article_embeddings.parquet` | OpenAI 1536d 임베딩 (406건, 약 5MB) | ✅ 5/27 |
| `data/keywords.parquet` | 검색 키워드 37건 | ✅ 5/27 |

> 6/4 최종 갱신 예정 (예상 1,500~2,000건)

---

## 2. 스키마

### `insk_corpus.parquet` (articles)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| article_id | BIGINT (PK) | 고유 ID |
| title | VARCHAR(255) | 한국어 95%+ (영문 모델·회사명 일부 혼합) |
| original_url | TEXT | 원본 기사 URL |
| published_at | DATETIME | 매체 발행 시각 (§4.6 참조: AITimes/TheGuru 60+60건은 부정확) |
| created_at | DATETIME | INSK 수집 시각 |
| source | VARCHAR(100) | "Naver" / "AITimes" / "TheGuru" |
| country | VARCHAR(10) | 거의 "KR" |
| language | VARCHAR(10) | 거의 "ko" |

> ⚠️ **body 컬럼 없음**: INSK는 본문을 OpenAI 분석에 사용 후 저장하지 않음. retrieval에 사용 가능한 텍스트는 `title` + `article_analyses.summary` (5줄) + `article_analyses.insight` (1줄) + `article_analyses.tags`. 벡터 검색은 `article_embeddings`(본문 기반으로 사전 계산됨)로 가능 → §4.7 참조.

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
| embedding_id (PK) | DB에서는 `id` 컬럼, parquet에서는 `embedding_id`로 rename |
| article_id (FK) | |
| embedding_json | **1536차원 float**, JSON 문자열 직렬화. 본문 6000자 truncate 후 OpenAI 호출 결과 |

**로드 예시**:
```python
import json
import numpy as np

df = pd.read_parquet("data/article_embeddings.parquet")
df["embedding"] = df["embedding_json"].apply(lambda s: np.array(json.loads(s)))
# df["embedding"].iloc[0].shape → (1536,)
```

### `keywords.parquet` (검색 키워드 37개, 2026-05-27 기준)

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

### 4.1 분류 편향 (mitigated, 일부 후퇴)

**현황 (2026-05-27 실측, 누적 406건)**:
| 카테고리 | 5/22 마이그레이션 직후 | 5/27 현재 | 변화 |
|---|:---:|:---:|---|
| AI Business | 55% | **55.7%** | 유지 |
| LLM | **20%** | **14.8%** | ⚠️ 5.2%p 후퇴 |
| INFRA | - | 15.8% | |
| Telco | - | 13.8% | |

→ 신규 trigger에서 LLM 비율이 다시 떨어짐. 신규 86건만 별도 분석 + SYSTEM_PROMPT 추가 보강 검토 중 (W3 박건우 작업).

**영향**:
- ❌ retrieval 실험 (Embedding / Reranker)은 임베딩 기반 → 영향 없음
- ⚠️ BM25는 본문이 없어서 title+summary 기반으로 해야 함 (§4.7 참조)
- ✅ category 컬럼 필터링은 LLM 부족할 수 있음 → tags·summary 매칭으로 우회 권장

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

### 4.4 매체 편향 (5/27 실측, 심화 추세)

| 매체 | 5/22 | 5/27 | 비고 |
|---|:---:|:---:|---|
| Naver | 60% | **70.4%** (286건) | search API 풍부, 누적할수록 비율 상승 |
| AITimes | 18% | 14.8% (60건) | RSS 매일 1-2건 갱신 천장 |
| TheGuru | 18% | 14.8% (60건) | 동일 |

→ retrieval 실험 시 매체별 균형 sampling 평가 권장. 6/4 시점에는 Naver 75-80% 예상.

### 4.5 외부 매체 검색 한계

- Naver `display=10` + `sort=sim` + `start` 없음 → 키워드당 최대 10건 풀
- AITimes/TheGuru RSS는 매체 최신 발행분만
- **과거 데이터 백필 불가능** → 시계열 길이는 INSK 가동 시작 시점부터

### 4.6 ⚠️ AITimes / TheGuru published_at 부정확 (2026-05-27 발견)

- 기존 AITimes 60건 + TheGuru 60건의 `published_at`은 INSK 수집 시각(`created_at`)과 동일
- 코드 버그: `processAITimes` / `processTheGuru`에서 RSS pubDate 파싱 안 하고 `LocalDateTime.now()` 사용
- 2026-05-27 INSK commit `0033b84`로 코드 수정 완료, **신규 trigger 분부터 정확한 매체 발행 시각**
- 기존 120건은 RSS lookback 불가라 backfill 못 함
- **영향**: temporal weighting 실험 시 AITimes/TheGuru 120건은 published_at 대신 source filter로 제외하거나 별도 처리 권장

### 4.7 🚨 articles.body 없음 (가장 중요)

INSK는 본문을 OpenAI 분석 입력으로 한 번 쓴 뒤 저장하지 않음. retrieval 실험에서 사용 가능한 텍스트:

| 텍스트 | 출처 | 평균 길이 |
|---|---|---|
| title | `insk_corpus.title` | 30~80자 |
| summary | `article_analyses.summary` | 5줄 (약 250자) |
| insight | `article_analyses.insight` | 1줄 (약 80자) |
| tags | `article_analyses.tags` | 3 keywords |
| **embedding** | `article_embeddings.embedding_json` | **본문 6000자 truncate 기반 1536d 벡터** |

**팀원별 영향**:
- 팀원 A (BM25): title + summary + insight + tags concat 해서 인덱스 만들기
- 팀원 A (Embedding 검색): `article_embeddings` 그대로 사용 가능 ✅
- 팀원 B (Reranker): 쿼리·문서 쌍 학습 시 문서 = title + summary 형태
- 팀원 C (RAG 답변): LLM에 넣을 context는 title + summary + insight

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
