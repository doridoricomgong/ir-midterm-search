from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ir_engine import InauguralSearchEngine


st.set_page_config(page_title="IR 중간고사 검색 시스템", layout="wide")

BM25_K1 = 1.2
BM25_B = 0.9
SAMPLE_QUERIES = ["freedom union", "war peace", "constitution people"]


@st.cache_resource(show_spinner="NLTK Inaugural Address 코퍼스를 색인하는 중입니다.")
def get_engine() -> InauguralSearchEngine:
    documents = InauguralSearchEngine.load_inaugural_documents(stop_year=2021)
    return InauguralSearchEngine(documents)


def make_doc_url(file_id: str) -> str:
    return f"?doc={file_id}"


def inject_design_system() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,750&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

        :root {
            --ink: #17221c;
            --moss: #2f513d;
            --sage: #7d927f;
            --paper: #f7f0df;
            --paper-deep: #eadcbd;
            --line: rgba(45, 55, 44, 0.18);
            --red: #a33b2f;
            --gold: #b48432;
            --shadow: 0 24px 70px rgba(42, 36, 24, 0.16);
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 9% 6%, rgba(179, 132, 50, 0.28), transparent 25rem),
                radial-gradient(circle at 92% 2%, rgba(47, 81, 61, 0.22), transparent 26rem),
                linear-gradient(135deg, rgba(23, 34, 28, 0.05) 25%, transparent 25%) 0 0 / 24px 24px,
                linear-gradient(180deg, #fbf7ed 0%, #ece2c9 100%);
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: rgba(247, 240, 223, 0.86);
            border-right: 1px solid var(--line);
            backdrop-filter: blur(14px);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.2rem;
        }

        h1, h2, h3, .serif {
            font-family: "Fraunces", Georgia, serif !important;
            letter-spacing: -0.035em;
        }

        p, div, label, input, textarea, button {
            font-family: "Source Serif 4", Georgia, serif;
        }

        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 34px;
            padding: 34px 36px;
            margin-bottom: 22px;
            background:
                linear-gradient(120deg, rgba(255,255,255,0.62), rgba(234,220,189,0.74)),
                repeating-linear-gradient(90deg, transparent 0 42px, rgba(23,34,28,0.035) 42px 43px);
            box-shadow: var(--shadow);
        }

        .hero::after {
            content: "IR";
            position: absolute;
            right: -20px;
            top: -52px;
            font: 750 190px "Fraunces", Georgia, serif;
            color: rgba(163, 59, 47, 0.08);
            transform: rotate(-9deg);
        }

        .eyebrow {
            display: inline-flex;
            gap: 10px;
            align-items: center;
            padding: 8px 12px;
            border: 1px solid rgba(163, 59, 47, 0.28);
            border-radius: 999px;
            background: rgba(163, 59, 47, 0.08);
            color: #7c2d25;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.76rem;
        }

        .hero-title {
            margin: 18px 0 10px;
            max-width: 850px;
            font: 750 clamp(2.4rem, 6vw, 5.6rem)/0.92 "Fraunces", Georgia, serif;
            letter-spacing: -0.075em;
            color: var(--ink);
        }

        .hero-copy {
            max-width: 720px;
            font-size: 1.08rem;
            line-height: 1.58;
            color: rgba(23, 34, 28, 0.78);
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 22px;
        }

        .stat-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 18px 18px 16px;
            background: rgba(255, 252, 243, 0.72);
            box-shadow: 0 12px 30px rgba(42, 36, 24, 0.08);
        }

        .stat-label {
            color: rgba(23, 34, 28, 0.58);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .stat-value {
            margin-top: 6px;
            font: 750 1.9rem "Fraunces", Georgia, serif;
            color: var(--moss);
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 22px;
            background: rgba(255, 252, 243, 0.74);
            box-shadow: 0 16px 44px rgba(42, 36, 24, 0.10);
        }

        .section-title {
            margin: 8px 0 14px;
            font: 750 1.45rem "Fraunces", Georgia, serif;
            color: var(--ink);
        }

        .method-note {
            border-left: 5px solid var(--red);
            border-radius: 18px;
            padding: 15px 17px;
            background: rgba(163, 59, 47, 0.08);
            color: rgba(23, 34, 28, 0.78);
            line-height: 1.5;
        }

        .result-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(47, 81, 61, 0.22);
            border-radius: 28px;
            margin: 16px 0;
            padding: 22px 24px 20px 92px;
            background:
                linear-gradient(90deg, rgba(47, 81, 61, 0.12), transparent 28%),
                rgba(255, 252, 243, 0.84);
            box-shadow: 0 18px 44px rgba(42, 36, 24, 0.12);
            animation: result-in 420ms ease both;
        }

        @keyframes result-in {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .rank-stamp {
            position: absolute;
            left: 20px;
            top: 20px;
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            background: var(--ink);
            color: var(--paper);
            font: 750 1.1rem "Fraunces", Georgia, serif;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.16);
        }

        .result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }

        .president {
            font: 750 1.55rem "Fraunces", Georgia, serif;
            color: var(--ink);
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(47, 81, 61, 0.10);
            border: 1px solid rgba(47, 81, 61, 0.18);
            color: var(--moss);
            font-weight: 700;
            font-size: 0.84rem;
        }

        .score-pill {
            background: rgba(180, 132, 50, 0.18);
            border-color: rgba(180, 132, 50, 0.26);
            color: #74511c;
        }

        .snippet {
            margin: 12px 0 14px;
            padding-left: 16px;
            border-left: 3px solid rgba(163, 59, 47, 0.45);
            color: rgba(23, 34, 28, 0.82);
            font-size: 1.03rem;
            line-height: 1.56;
        }

        .tf-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .tf-token {
            padding: 7px 10px;
            border-radius: 12px;
            background: rgba(23, 34, 28, 0.06);
            border: 1px dashed rgba(23, 34, 28, 0.20);
            color: rgba(23, 34, 28, 0.78);
            font-size: 0.9rem;
            font-weight: 700;
        }

        .doc-link {
            display: inline-flex;
            margin-top: 12px;
            color: #7c2d25 !important;
            font-weight: 800;
            text-decoration: none !important;
            border-bottom: 2px solid rgba(163, 59, 47, 0.28);
        }

        .doc-link:hover {
            border-bottom-color: #7c2d25;
        }

        .checklist {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
            margin-top: 14px;
        }

        .check {
            padding: 10px 12px;
            border-radius: 14px;
            background: rgba(47, 81, 61, 0.09);
            color: rgba(23, 34, 28, 0.78);
            font-weight: 700;
            font-size: 0.9rem;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 18px;
            border: 1px solid rgba(47, 81, 61, 0.22);
            background: rgba(255, 252, 243, 0.88);
            font-size: 1.08rem;
            padding: 0.85rem 1rem;
        }

        div[data-testid="stRadio"] {
            padding: 12px 14px;
            border-radius: 18px;
            background: rgba(255, 252, 243, 0.58);
            border: 1px solid var(--line);
        }

        .doc-view {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 24px;
            background: rgba(255, 252, 243, 0.82);
            box-shadow: var(--shadow);
        }

        @media (max-width: 860px) {
            .stat-grid, .checklist { grid-template-columns: 1fr; }
            .hero { padding: 26px 22px; }
            .result-card { padding-left: 22px; padding-top: 84px; }
            .rank-stamp { top: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_document_view(engine: InauguralSearchEngine, file_id: str) -> None:
    inject_design_system()
    document = next((doc for doc in engine.documents if doc.file_id == file_id), None)
    if document is None:
        st.error("요청한 취임사를 찾을 수 없습니다.")
        return

    st.markdown(
        f"""
        <div class="doc-view">
          <div class="eyebrow">Primary Source</div>
          <div class="hero-title" style="font-size: clamp(2.0rem, 5vw, 4.4rem);">
            {document.year} {escape(document.president)}
          </div>
          <p class="hero-copy">file id: <strong>{escape(document.file_id)}</strong> / doc id: <strong>{document.doc_id}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.link_button("검색 화면으로 돌아가기", "./")
    st.text_area("취임사 원문", document.text, height=560)


def render_query_diagnostics(engine: InauguralSearchEngine, query: str) -> None:
    idf_rows = [
        {"term": term, "idf": round(idf, 6)}
        for term, idf in engine.query_idf(query).items()
    ]
    st.markdown("### 질의어 IDF")
    if idf_rows:
        st.dataframe(pd.DataFrame(idf_rows), width="stretch", hide_index=True)
    else:
        st.info("질의어를 입력하면 각 term의 IDF가 표시됩니다.")


def render_results(engine: InauguralSearchEngine, query: str, method: str, top_k: int) -> None:
    results = engine.search(query, method=method, top_k=top_k)
    if not results:
        st.warning("검색 결과가 없습니다. 다른 질의어를 입력하세요.")
        return

    st.markdown('<div class="section-title">Ranked Addresses</div>', unsafe_allow_html=True)
    for rank, result in enumerate(results, start=1):
        tf_html = "".join(
            f'<span class="tf-token">{escape(term)} TF={tf}</span>'
            for term, tf in result.term_tf.items()
        )
        st.markdown(
            f"""
            <article class="result-card" style="animation-delay: {min(rank * 35, 280)}ms;">
              <div class="rank-stamp">#{rank}</div>
              <div class="result-meta">
                <span class="president">{result.year} {escape(result.president)}</span>
                <span class="pill score-pill">{escape(method)} score {result.score:.4f}</span>
                <span class="pill">doc {result.doc_id}</span>
                <span class="pill">{escape(result.file_id)}</span>
              </div>
              <div class="snippet">"{escape(result.snippet)}"</div>
              <div class="tf-strip">{tf_html}</div>
              <a class="doc-link" href="{make_doc_url(result.file_id)}">취임사 원문 열기</a>
            </article>
            """,
            unsafe_allow_html=True,
        )


def render_search_view(engine: InauguralSearchEngine) -> None:
    inject_design_system()
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Information Retrieval Midterm</div>
          <div class="hero-title">Presidential Address Search Lab</div>
          <p class="hero-copy">
            NLTK 미국 대통령 취임사 코퍼스를 색인하고, cosine TF-IDF와
            BM25(k1={BM25_K1}, b={BM25_B})의 결과 차이를 제출용으로 바로 확인합니다.
          </p>
          <div class="checklist">
            <div class="check">docID + 대통령명 저장</div>
            <div class="check">tokenize + stemming</div>
            <div class="check">dictionary + inverted index</div>
            <div class="check">TF/IDF 노출</div>
            <div class="check">cosine / BM25 랭킹</div>
            <div class="check">중요 문장 snippet</div>
          </div>
        </section>
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-label">Corpus</div><div class="stat-value">{engine.document_count}</div></div>
          <div class="stat-card"><div class="stat-label">Dictionary</div><div class="stat-value">{len(engine.dictionary):,}</div></div>
          <div class="stat-card"><div class="stat-label">Avg Length</div><div class="stat-value">{engine.average_document_length:.0f}</div></div>
          <div class="stat-card"><div class="stat-label">Range</div><div class="stat-value">1789-2021</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Search Console</div>', unsafe_allow_html=True)
    initial_query = st.query_params.get("q", "freedom union")
    query = st.text_input("검색어", value=initial_query, placeholder="예: freedom union")

    sample_cols = st.columns(len(SAMPLE_QUERIES))
    for col, sample in zip(sample_cols, SAMPLE_QUERIES):
        if col.button(sample, width="stretch"):
            st.query_params["q"] = sample
            st.rerun()

    control_cols = st.columns([1, 1, 2])
    with control_cols[0]:
        method = st.radio("Ranking method", ["cosine", "BM25"], horizontal=True)
    with control_cols[1]:
        top_k = st.slider("결과 수", min_value=3, max_value=20, value=10)
    with control_cols[2]:
        st.markdown(
            f"""
            <div class="method-note">
              <strong>{escape(method)}</strong> 선택 중입니다.
              cosine은 TF-IDF 벡터 각도, BM25는 문서 길이 보정 TF를 사용합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not query.strip():
        st.info("검색어를 입력하세요.")
        return

    diag_col, guide_col = st.columns([1.05, 1.0])
    with diag_col:
        render_query_diagnostics(engine, query)
    with guide_col:
        st.markdown(
            """
            <div class="panel">
              <div class="section-title">읽는 방법</div>
              <p class="hero-copy" style="font-size: 0.98rem;">
                각 결과 카드는 대통령/연도, 랭킹 점수, 질의어 TF, 그리고 질의어 IDF 기반으로
                고른 대표 문장을 함께 보여줍니다. 원문 링크는 Streamlit 내부 URL입니다.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_results(engine, query, method, top_k)


def main() -> None:
    engine = get_engine()
    requested_file_id = st.query_params.get("doc")
    if requested_file_id:
        render_document_view(engine, requested_file_id)
        return
    render_search_view(engine)


if __name__ == "__main__":
    main()
