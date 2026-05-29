# -*- coding: utf-8 -*-
"""
Query Characteristic Analysis  (OpenAI 포함 5모델 버전)
Team A Core Contribution: per-query-characteristic model strength analysis

Models: BM25 / OpenAI(text-embedding-3-small) / BGE-M3 / ko-sroberta / Hybrid(RRF)

Characteristics:
  - style   : lexical (keyword/entity match) / semantic (conceptual meaning)
  - has_eng : True = contains English technical terms (OpenAI, HBM, NVIDIA...)
  - has_kor : True = contains Korean proper nouns / company names (삼성전자, SKT...)
"""

import io, json, math, os, re, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print("=== 데이터 로드 ===")
corpus   = pd.read_parquet(DATA / "insk_corpus.parquet")
analyses = pd.read_parquet(DATA / "article_analyses.parquet")
emb_df   = pd.read_parquet(DATA / "article_embeddings.parquet")

df = corpus.merge(analyses, on="article_id", how="left")

def parse_tags(t):
    if pd.isna(t): return ""
    try:
        lst = json.loads(t) if isinstance(t, str) else t
        return " ".join(lst) if isinstance(lst, list) else str(lst)
    except Exception:
        return str(t)

df["tags_str"] = df["tags"].apply(parse_tags) if "tags" in df.columns else ""
df["text"] = (
    df["title"].fillna("") + " " +
    df["summary"].fillna("") + " " +
    df["insight"].fillna("") + " " +
    df["tags_str"]
)

qa_data = []
with open(DATA / "human_qa_benchmark_v1.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            qa_data.append(json.loads(line))

all_article_ids = df["article_id"].tolist()
all_texts       = df["text"].tolist()
print(f"corpus: {len(df)}개, QA: {len(qa_data)}개")

# ── 2. BM25 ───────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text.lower())

bm25 = BM25Okapi([tokenize(t) for t in all_texts])
print("BM25 구축 완료")

# ── 3. OpenAI corpus 임베딩 로드 (article_embeddings.parquet) ──────────────────
print("\n=== OpenAI corpus 임베딩 로드 ===")

def parse_embedding(s):
    if pd.isna(s): return None
    try:
        parsed = json.loads(s)
        if parsed is None or not isinstance(parsed, list): return None
        return np.array(parsed, dtype=np.float32)
    except Exception:
        return None

emb_series = emb_df.set_index("article_id")["embedding_json"].apply(parse_embedding)
valid_emb  = {aid: e for aid, e in emb_series.items() if e is not None}

# corpus 순서에 맞게 정렬 (없는 건 None)
openai_E_list = []
openai_valid_ids = []
for aid in all_article_ids:
    e = valid_emb.get(aid)
    if e is not None:
        openai_E_list.append(e)
        openai_valid_ids.append(aid)

openai_E      = np.vstack(openai_E_list).astype(np.float32)
norms         = np.linalg.norm(openai_E, axis=1, keepdims=True)
norms[norms == 0] = 1e-9
openai_E_norm = openai_E / norms
print(f"OpenAI corpus 임베딩 shape: {openai_E_norm.shape}  (null 제외: {len(all_article_ids)-len(openai_valid_ids)}개 제외)")

# OpenAI query 인코딩 함수
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
if OPENAI_AVAILABLE:
    from openai import OpenAI
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI 클라이언트 초기화 완료")
else:
    print("OpenAI API 키 없음 → OpenAI 모델 제외")

def openai_retrieve(query, top_k=10):
    resp  = oa_client.embeddings.create(model="text-embedding-3-small", input=[query])
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims  = openai_E_norm @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [openai_valid_ids[i] for i in top_idx]

# ── 4. SentenceTransformer 모델 로드 & corpus 인코딩 ──────────────────────────
print("\n=== SentenceTransformer 모델 로드 & corpus 인코딩 ===")
ST_AVAILABLE = False
bge_E_norm = ksr_E_norm = bge_model = ksr_model = None

try:
    from sentence_transformers import SentenceTransformer

    print("BGE-M3 로드 중...")
    bge_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print("BGE-M3 corpus 인코딩 중...")
    bge_E = bge_model.encode(all_texts, normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True).astype(np.float32)
    norms = np.linalg.norm(bge_E, axis=1, keepdims=True); norms[norms==0] = 1e-9
    bge_E_norm = bge_E / norms
    print(f"BGE-M3 shape: {bge_E_norm.shape}")

    print("ko-sroberta 로드 중...")
    ksr_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cpu")
    print("ko-sroberta corpus 인코딩 중...")
    ksr_E = ksr_model.encode(all_texts, normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True).astype(np.float32)
    norms = np.linalg.norm(ksr_E, axis=1, keepdims=True); norms[norms==0] = 1e-9
    ksr_E_norm = ksr_E / norms
    print(f"ko-sroberta shape: {ksr_E_norm.shape}")

    ST_AVAILABLE = True
    print("SentenceTransformer 완료")
except Exception as e:
    print(f"SentenceTransformer 오류: {e}")

# ── 5. 검색 함수 ───────────────────────────────────────────────────────────────
def bm25_retrieve(query, top_k=10):
    scores  = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [all_article_ids[i] for i in top_idx]

def dense_retrieve(query, model, E_norm, id_list, top_k=10):
    q_emb  = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    sims   = E_norm @ q_norm
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [id_list[i] for i in top_idx]

def rrf_merge(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

def hybrid_retrieve(query, top_k=10):
    ranked = [bm25_retrieve(query, 50)]
    if ST_AVAILABLE:
        ranked.append(dense_retrieve(query, bge_model, bge_E_norm, all_article_ids, 50))
    return rrf_merge(ranked)[:top_k]

# ── 6. 평가 지표 ───────────────────────────────────────────────────────────────
def recall_at_k(retrieved, gold, k=5):
    if not gold: return None
    return sum(1 for r in retrieved[:k] if r in gold) / min(len(gold), k)

def mrr_score(retrieved, gold):
    if not gold: return None
    for i, r in enumerate(retrieved, 1):
        if r in gold: return 1.0 / i
    return 0.0

# ── 7. QA 특성 태깅 ────────────────────────────────────────────────────────────
QA_CHAR = {
    "L1": {"style": "lexical",  "has_eng": True,  "has_kor": False},
    "L2": {"style": "semantic", "has_eng": False, "has_kor": False},
    "L3": {"style": "lexical",  "has_eng": True,  "has_kor": False},
    "L4": {"style": "lexical",  "has_eng": True,  "has_kor": True },
    "L5": {"style": "semantic", "has_eng": False, "has_kor": False},
    "L6": {"style": "lexical",  "has_eng": True,  "has_kor": False},
    "I1": {"style": "lexical",  "has_eng": False, "has_kor": True },
    "I2": {"style": "lexical",  "has_eng": True,  "has_kor": True },
    "I3": {"style": "lexical",  "has_eng": True,  "has_kor": False},
    "I4": {"style": "semantic", "has_eng": False, "has_kor": False},
    "I5": {"style": "semantic", "has_eng": False, "has_kor": False},
    "I6": {"style": "semantic", "has_eng": True,  "has_kor": False},
    "B1": {"style": "lexical",  "has_eng": True,  "has_kor": False},
    "B2": {"style": "semantic", "has_eng": True,  "has_kor": False},
    "B3": {"style": "semantic", "has_eng": False, "has_kor": False},
    "B4": {"style": "semantic", "has_eng": False, "has_kor": False},
    "B5": {"style": "lexical",  "has_eng": False, "has_kor": True },
    "B6": {"style": "semantic", "has_eng": False, "has_kor": False},
    "B7": {"style": "semantic", "has_eng": False, "has_kor": False},
    "B8": {"style": "semantic", "has_eng": False, "has_kor": False},
    "B9": {"style": "semantic", "has_eng": False, "has_kor": False},
    "T1": {"style": "lexical",  "has_eng": True,  "has_kor": True },
    "T2": {"style": "lexical",  "has_eng": False, "has_kor": True },
    "T3": {"style": "lexical",  "has_eng": True,  "has_kor": True },
    "T4": {"style": "semantic", "has_eng": True,  "has_kor": False},
    "T5": {"style": "lexical",  "has_eng": False, "has_kor": False},
    "T6": {"style": "semantic", "has_eng": False, "has_kor": False},
}

# ── 8. 실험 실행 ───────────────────────────────────────────────────────────────
print("\n=== QA 실험 실행 중 ===")

COLS_R5 = ["BM25_R5", "OAI_R5", "BGE_R5", "KSR_R5", "HYB_R5"]
if not OPENAI_AVAILABLE: COLS_R5.remove("OAI_R5")
if not ST_AVAILABLE:
    for c in ["BGE_R5", "KSR_R5"]:
        if c in COLS_R5: COLS_R5.remove(c)

LABEL = {
    "BM25_R5": "BM25",
    "OAI_R5":  "OpenAI",
    "BGE_R5":  "BGE-M3",
    "KSR_R5":  "ko-sroberta",
    "HYB_R5":  "Hybrid",
}

results = []

for qa in qa_data:
    qid      = qa["id"]
    q        = qa["question"]
    gold     = set(qa["gold_articles"])
    cat      = qa.get("category", "unknown")
    qtype_tag= qa.get("type", "unknown")
    char     = QA_CHAR.get(qid, {"style": "unknown", "has_eng": False, "has_kor": False})
    is_neg   = (len(gold) == 0)

    row = {
        "qid": qid, "question": q, "category": cat,
        "qtype": qtype_tag, "style": char["style"],
        "has_eng": char["has_eng"], "has_kor": char["has_kor"],
        "is_neg": is_neg,
    }

    b_ret = bm25_retrieve(q, 10)
    row["BM25_R5"] = recall_at_k(b_ret, gold)

    if OPENAI_AVAILABLE:
        o_ret = openai_retrieve(q, 10)
        row["OAI_R5"] = recall_at_k(o_ret, gold)

    if ST_AVAILABLE:
        bge_ret = dense_retrieve(q, bge_model, bge_E_norm, all_article_ids, 10)
        ksr_ret = dense_retrieve(q, ksr_model, ksr_E_norm, all_article_ids, 10)
        row["BGE_R5"] = recall_at_k(bge_ret, gold)
        row["KSR_R5"] = recall_at_k(ksr_ret, gold)

    h_ret = hybrid_retrieve(q, 10)
    row["HYB_R5"] = recall_at_k(h_ret, gold)

    results.append(row)

    neg_mark = " [Neg]" if is_neg else ""
    vals = "  ".join(
        f"{LABEL[c]}={'N/A' if row.get(c) is None else f'{row[c]:.2f}'}"
        for c in COLS_R5
    )
    print(f"  {qid:<3} ({char['style'][:3]})  {vals}  gold={len(gold)}{neg_mark}")

res_df  = pd.DataFrame(results)
pos_df  = res_df[res_df["is_neg"] == False].copy()
for c in COLS_R5:
    pos_df[c] = pd.to_numeric(pos_df[c], errors="coerce")

SEP = "=" * 72

# ── 9. 분석 1: Lexical vs Semantic ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  분석 1: Lexical vs Semantic 스타일별 Recall@5")
print(SEP)

style_mean = pos_df.groupby("style")[COLS_R5].mean()
style_mean.columns = [LABEL[c] for c in COLS_R5]
print(style_mean.round(3).to_string())
print()
for style in ["lexical", "semantic"]:
    if style not in style_mean.index: continue
    row_s = style_mean.loc[style].dropna()
    best  = row_s.idxmax()
    n     = len(pos_df[pos_df["style"] == style])
    print(f"  [{style.upper():8s}] {n}개 쿼리 --> 최적 모델: {best} (Recall@5={row_s[best]:.3f})")

# ── 10. 분석 2: 한국어 고유명사 vs 영문 기술용어 ──────────────────────────────────
print(f"\n{SEP}")
print("  분석 2: 한국어 고유명사 vs 영문 기술용어 Recall@5")
print(SEP)

kor_df = pos_df[pos_df["has_kor"] == True]
eng_df = pos_df[pos_df["has_eng"] == True]
nei_df = pos_df[(pos_df["has_kor"] == False) & (pos_df["has_eng"] == False)]

def print_subset(subset, label):
    if len(subset) == 0: return
    avg  = subset[COLS_R5].mean()
    best = LABEL[avg.dropna().idxmax()]
    vals = "  ".join(f"{LABEL[c]}={avg[c]:.3f}" for c in COLS_R5 if not pd.isna(avg[c]))
    print(f"  {label:<22} (n={len(subset):2d})  {vals}")
    print(f"    >> 최적 모델: {best}")

print_subset(kor_df, "한국어 고유명사 포함")
print_subset(eng_df, "영문 기술용어 포함")
print_subset(nei_df, "언어 특성 없음(일반 의미론)")

# ── 11. 분석 3: QA 타입별 ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  분석 3: QA 타입별 Recall@5 (Strict / Trend / Negative)")
print(SEP)

all_type = res_df.copy()
for c in COLS_R5:
    all_type[c] = pd.to_numeric(all_type[c], errors="coerce")

type_mean = all_type.groupby("qtype")[COLS_R5].mean()
type_mean.columns = [LABEL[c] for c in COLS_R5]
print(type_mean.round(3).to_string())

# ── 12. 분석 4: Category별 ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  분석 4: Category별 Recall@5 (Negative 제외)")
print(SEP)

cat_mean = pos_df.groupby("category")[COLS_R5].mean()
cat_mean.columns = [LABEL[c] for c in COLS_R5]
print(cat_mean.round(3).to_string())

# ── 13. 분석 5: Per-QA 최적 모델 ─────────────────────────────────────────────
print(f"\n{SEP}")
print("  분석 5: QA별 최적 모델 (Recall@5 기준, Negative 제외)")
print(SEP)

model_labels = [LABEL[c] for c in COLS_R5]
col_w = 13

hdr = f"{'QID':<4} {'Type':<9} {'Style':<8} {'KOR':<4} {'ENG':<4} "
hdr += " ".join(f"{lb:<{col_w}}" for lb in model_labels) + " Best"
print(hdr)
print("-" * (len(hdr) + 4))

best_count = {lb: 0 for lb in model_labels}

for _, r in pos_df.iterrows():
    scores = {LABEL[c]: r[c] for c in COLS_R5 if pd.notna(r.get(c))}
    if not scores: continue
    best_model = max(scores, key=scores.get)
    best_count[best_model] += 1
    vals = " ".join(f"{scores.get(lb, float('nan')):<{col_w}.3f}" for lb in model_labels)
    print(f"{r['qid']:<4} {r['qtype']:<9} {r['style'][:3]:<8} "
          f"{'Y' if r['has_kor'] else 'N':<4} {'Y' if r['has_eng'] else 'N':<4} "
          f"{vals} {best_model}")

n_pos = len(pos_df)
print(f"\n  [최적 모델 집계] (총 {n_pos}개 Positive QA)")
for m, cnt in sorted(best_count.items(), key=lambda x: -x[1]):
    print(f"    {m:<14} {'#'*cnt} {cnt}회")

# ── 14. 최종 인사이트 요약 ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ★ 핵심 인사이트 요약")
print(SEP)

overall = pos_df[COLS_R5].mean()
best_overall = LABEL[overall.dropna().idxmax()]

print(f"\n  [전체 평균 Recall@5] (Positive {n_pos}개 기준)")
for c in COLS_R5:
    marker = " <-- 최고" if LABEL[c] == best_overall else ""
    print(f"    {LABEL[c]:<14}: {overall[c]:.3f}{marker}")

print("\n  [Query 스타일별 권장 모델]")
for style in ["lexical", "semantic"]:
    if style in style_mean.index:
        row_s = style_mean.loc[style].dropna()
        best  = row_s.idxmax()
        n_q   = len(pos_df[pos_df["style"] == style])
        print(f"    {style.upper():<9} ({n_q}개) --> {best:<14} Recall@5={row_s[best]:.3f}")

print("\n  [언어 특성별 권장 모델]")
for subset, label in [(kor_df,"한국어 고유명사"),(eng_df,"영문 기술용어"),(nei_df,"일반 의미론")]:
    if len(subset) > 0:
        avg  = subset[COLS_R5].mean().dropna()
        best = LABEL[avg.idxmax()]
        n_q  = len(subset)
        print(f"    {label:<12} ({n_q}개) --> {best:<14} Recall@5={avg.max():.3f}")

print("\n  [Category별 권장 모델]")
for cat in cat_mean.index:
    row_c = cat_mean.loc[cat].dropna()
    best  = row_c.idxmax()
    n_q   = len(pos_df[pos_df["category"] == cat])
    print(f"    {cat:<14} ({n_q}개) --> {best:<14} Recall@5={row_c[best]:.3f}")

print(f"\n{SEP}")
print("  분석 완료")
print(SEP)
