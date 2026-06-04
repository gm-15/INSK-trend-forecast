"""
INSK Trend Forecast — RAG 데모 (서비스 시연용)

중간발표 슬라이드 8 "실제 작동하는 서비스 시연" 충족.
INSK 운영 데이터(뉴스 corpus)로 팀 A·B·C 파이프라인을 라이브로 시연한다.

파이프라인 (팀 작업 그대로 재현):
  1. Hybrid Retrieval (BM25 + BGE-M3, RRF)  ← 팀 A
  2. Cross-encoder Reranker (klue/bert-base fine-tuned) + Temporal decay  ← 팀 B + C
  3. GPT-4o-mini 답변 생성 (근거 인용 + 환각 방지)  ← 팀 C

실행:
  cd C:\\dev\\INSK-trend-forecast
  $env:OPENAI_API_KEY = "sk-..."
  streamlit run app/streamlit_demo.py
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# 경로 설정
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS_PATH = DATA / "insk_corpus.parquet"
ANALYSIS_PATH = DATA / "article_analyses.parquet"
QA_PATH = DATA / "human_qa_benchmark_v1.jsonl"

# reranker 모델 후보 경로 (445MB, git 제외 — 로컬/드라이브에서 탐색)
RERANKER_CANDIDATES = [
    ROOT / "src" / "reranker" / "reranker_ft.pt",
    Path(os.environ.get("RERANKER_PATH", "")) if os.environ.get("RERANKER_PATH") else None,
    Path.home() / "Downloads" / "src-20260530T160212Z-3-001" / "src" / "reranker" / "reranker_ft.pt",
]

# 팀 파이프라인 하이퍼파라미터 (노트북 09와 동일)
LAMBDA_DECAY = 0.05
BASE_DATE = pd.to_datetime("2026-05-23")
BGE_MODEL = "BAAI/bge-m3"
RERANKER_BASE = "klue/bert-base"
GEN_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "너는 한국어 AI 산업 뉴스 전문가이며, 철저하게 주어진 [Context]에만 기반하여 답변하는 챗봇이다. "
    "다음 규칙을 반드시 칼같이 지켜라:\n\n"
    "1. 할루시네이션(지어내기) 절대 금지:\n"
    "- 답변은 오직 제공된 [Context]의 텍스트에서 명시적으로 언급된 사실만 바탕으로 작성해야 한다.\n\n"
    "2. 답변 불가능 시 방어 대사 연결:\n"
    "- 관련 내용을 찾을 수 없으면 정확히 '제공된 뉴스에서 관련 내용을 찾을 수 없습니다.' 만 출력하라.\n\n"
    "3. 엄격한 출처 인용구 부착:\n"
    "- 모든 개별 문장 끝 마침표 바로 앞에 '[1]', '[2]' 형태의 인용구를 부착하라."
)


# ----------------------------------------------------------------------------
# 데이터 로드 (데이터 사전 §4.2·4.6·4.7 필터 반영)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="뉴스 corpus 로드 중...")
def load_corpus():
    corpus = pd.read_parquet(CORPUS_PATH)
    analysis = pd.read_parquet(ANALYSIS_PATH)
    df = pd.merge(corpus, analysis, on="article_id", how="inner")

    df["published_at"] = pd.to_datetime(df["published_at"])
    df = df[df["published_at"] >= "2026-05-19"]  # §4.2 시계열 lumpy
    # §4.6 published_at 버그 있는 옛 RSS 제외 (Naver 또는 5/27 이후 수집분)
    df["created_at_x"] = pd.to_datetime(df["created_at_x"])
    df = df[(df["source"] == "Naver") | (df["created_at_x"] >= "2026-05-27")].copy()

    # §4.7 본문 없음 → title + summary + insight 결합 텍스트
    df["combined_text"] = (
        "제목: " + df["title"].fillna("")
        + "\n요약: " + df["summary"].fillna("")
        + "\n인사이트: " + df["insight"].fillna("")
    )
    df = df.reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_benchmark():
    qa = []
    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            qa.append(json.loads(line))
    return qa


# ----------------------------------------------------------------------------
# 모델 로드 (무거움 → cache_resource로 1회만)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="BGE-M3 임베딩 모델 로드 중... (최초 1회 다운로드)")
def load_bge():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BGE_MODEL)


@st.cache_resource(show_spinner="BM25 인덱스 구축 중...")
def build_bm25(_texts):
    from rank_bm25 import BM25Okapi
    tokenized = [t.split() for t in _texts]
    return BM25Okapi(tokenized)


EMB_CACHE = DATA / "corpus_bge_m3.npz"


@st.cache_resource(show_spinner="corpus 임베딩 로드 중...")
def embed_corpus(_texts, _article_ids):
    """corpus 임베딩. 파일 캐시(npz)가 현재 corpus와 일치하면 즉시 로드, 아니면 계산 후 저장.
    데모 cold-start 방지: 한 번 계산하면 data/corpus_bge_m3.npz에 저장됨."""
    ids = np.asarray(_article_ids)
    if EMB_CACHE.exists():
        cached = np.load(EMB_CACHE, allow_pickle=False)
        if cached["article_ids"].shape == ids.shape and np.array_equal(cached["article_ids"], ids):
            return cached["embeddings"]
    # 캐시 없음/불일치 → 계산 후 저장
    bge = load_bge()
    emb = np.asarray(bge.encode(list(_texts), normalize_embeddings=True, show_progress_bar=False))
    np.savez(EMB_CACHE, embeddings=emb, article_ids=ids)
    return emb


@st.cache_resource(show_spinner="Reranker(klue/bert-base) 로드 중...")
def load_reranker():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = next((str(p) for p in RERANKER_CANDIDATES if p and p.exists()), None)
    if model_path is None:
        return None, None  # 모델 없으면 reranker 단계 skip

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_BASE, num_labels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return tokenizer, model


# ----------------------------------------------------------------------------
# 검색 단계
# ----------------------------------------------------------------------------
def hybrid_retrieve(query, df, bm25, corpus_emb, top_k=10, rrf_k=60):
    """BM25 + BGE-M3 dense 검색을 RRF로 융합 → top_k article 인덱스"""
    # BM25 랭킹
    bm25_scores = bm25.get_scores(query.split())
    bm25_rank = np.argsort(bm25_scores)[::-1]

    # Dense 랭킹 (BGE-M3 cosine)
    bge = load_bge()
    q_emb = bge.encode([query], normalize_embeddings=True)[0]
    dense_scores = corpus_emb @ q_emb
    dense_rank = np.argsort(dense_scores)[::-1]

    # RRF 융합
    rrf = {}
    for rank, idx in enumerate(bm25_rank):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(dense_rank):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank)

    top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [int(i) for i, _ in top]


def rerank(query, df, cand_idx, tokenizer, model, top_n=3):
    """Cross-encoder 점수 × temporal decay → top_n. 모델 없으면 입력 순서 유지."""
    import torch

    rows = df.loc[cand_idx].copy()
    if tokenizer is None or model is None:
        rows["final_score"] = np.linspace(1.0, 0.5, len(rows))  # fallback: 순서 유지
    else:
        scores = []
        for _, row in rows.iterrows():
            inputs = tokenizer(
                query, row["combined_text"],
                return_tensors="pt", truncation=True, max_length=512,
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            scores.append(torch.sigmoid(logits).item())
        rows["semantic_score"] = scores
        days = (BASE_DATE - rows["published_at"]).dt.days.clip(lower=0)
        rows["temporal_weight"] = np.exp(-LAMBDA_DECAY * days)
        rows["final_score"] = rows["semantic_score"] * rows["temporal_weight"]

    rows = rows.sort_values("final_score", ascending=False)
    return rows.head(top_n)


def generate_answer(query, context_rows):
    """GPT-4o-mini로 근거 인용 답변 생성. 키 없으면 None."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    ctx = "\n\n".join(
        f"[{i+1}] {row['combined_text']}" for i, (_, row) in enumerate(context_rows.iterrows())
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"아래 제공된 [Context] 뉴스들을 읽고, [Question]에 대해 규칙을 지켜 답변해줘.\n\n[Context]\n{ctx}\n\n[Question]\n{query}"},
        ],
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="INSK RAG 데모", page_icon="📰", layout="wide")
st.title("📰 INSK Trend Forecast — RAG 시연")
st.caption("한국어 AI 뉴스 RAG: Hybrid 검색 → Reranker → 답변 생성. INSK 운영 데이터 기반.")

df = load_corpus()
qa_bench = load_benchmark()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    st.metric("검색 대상 기사 수", f"{len(df)}건")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    st.write("OpenAI 키:", "✅ 감지됨" if has_key else "❌ 없음 (검색까지만)")
    tok, rer = load_reranker()
    st.write("Reranker:", "✅ 로드됨" if rer is not None else "⚠️ 모델 없음 (검색 순서 유지)")

    st.divider()
    st.subheader("🎯 벤치마크 질문 (27개)")
    bench_q = st.selectbox(
        "고르면 아래 입력창에 채워집니다",
        ["(직접 입력)"] + [f"[{q['type'][:1]}] {q['question']}" for q in qa_bench],
    )
    top_k = st.slider("1차 검색 후보 수", 5, 20, 10)
    top_n = st.slider("리랭킹 후 사용할 문서 수", 1, 5, 3)

# 질문 입력
default_q = ""
gold_ids = None
if bench_q != "(직접 입력)":
    sel = qa_bench[[f"[{q['type'][:1]}] {q['question']}" for q in qa_bench].index(bench_q)]
    default_q = sel["question"]
    gold_ids = sel.get("gold_articles", [])

query = st.text_input("질문을 입력하세요", value=default_q, placeholder="예: 삼성전자 AI 반도체 전략은?")
run = st.button("🔍 검색 + 답변 생성", type="primary")

if run and query.strip():
    bm25 = build_bm25(tuple(df["combined_text"].tolist()))
    corpus_emb = embed_corpus(tuple(df["combined_text"].tolist()), tuple(df["article_id"].tolist()))

    # 1단계: Hybrid 검색
    with st.spinner("1단계: Hybrid 검색 (BM25 + BGE-M3)..."):
        cand_idx = hybrid_retrieve(query, df, bm25, corpus_emb, top_k=top_k)

    # 2단계: Reranking + Temporal
    with st.spinner("2단계: Reranker + Temporal..."):
        reranked = rerank(query, df, cand_idx, tok, rer, top_n=top_n)

    # 3단계: 답변 생성
    answer = None
    if has_key:
        with st.spinner("3단계: GPT-4o-mini 답변 생성..."):
            answer = generate_answer(query, reranked)

    # ---- 결과 표시 ----
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1️⃣ 1차 Hybrid 검색 결과")
        cand_df = df.loc[cand_idx, ["article_id", "title", "source", "published_at"]].copy()
        cand_df["순위"] = range(1, len(cand_df) + 1)
        if gold_ids:
            cand_df["정답?"] = cand_df["article_id"].apply(lambda x: "⭐" if x in gold_ids else "")
        st.dataframe(cand_df.set_index("순위"), use_container_width=True, height=380)

    with col2:
        st.subheader("2️⃣ Reranker + Temporal 재정렬")
        show = reranked[["article_id", "title", "source", "published_at"]].copy()
        show["순위"] = range(1, len(show) + 1)
        if "final_score" in reranked.columns:
            show["점수"] = reranked["final_score"].round(3).values
        if gold_ids:
            show["정답?"] = show["article_id"].apply(lambda x: "⭐" if x in gold_ids else "")
        st.dataframe(show.set_index("순위"), use_container_width=True, height=380)

    st.divider()
    st.subheader("3️⃣ 최종 답변 (근거 인용)")
    if answer:
        st.success(answer)
    elif has_key:
        st.info("답변 생성 결과가 비었습니다.")
    else:
        st.warning("OPENAI_API_KEY가 없어 답변 생성을 건너뜀. 검색·리랭킹까지만 시연됩니다.")

    # 근거 문서 펼치기
    with st.expander("📄 답변 근거로 쓰인 문서 보기"):
        for i, (_, row) in enumerate(reranked.iterrows()):
            st.markdown(f"**[{i+1}] {row['title']}**  ·  {row['source']}  ·  {row['published_at'].date()}")
            st.caption(row["combined_text"])

    if gold_ids:
        st.divider()
        hit = any(aid in gold_ids for aid in reranked["article_id"].tolist())
        st.write(f"**벤치마크 정답 article_id**: {gold_ids}  →  리랭킹 top-{top_n} 안에 정답 포함: {'✅' if hit else '❌'}")

elif run:
    st.error("질문을 입력하세요.")
