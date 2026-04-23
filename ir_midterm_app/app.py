from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from ir_engine import InauguralSearchEngine


st.set_page_config(page_title="IR 중간고사 검색 시스템", layout="wide")

BM25_K1 = 1.2
BM25_B = 0.9
SAMPLE_QUERIES = ["freedom union", "war peace", "constitution people"]
GITHUB_URL = "https://github.com/doridoricomgong/ir-midterm-search"
DEMO_URL = "https://ir-midterm-search.onrender.com"

PROGRAM_CHECKS = [
    {
        "requirement": "1. 문서 ID 부여 및 대통령 이름 저장",
        "implementation": "NLTK fileid 순서로 doc_id를 부여하고 파일명에서 연도/대통령명을 분리합니다.",
        "evidence": "결과 카드의 doc 번호, 연도, 대통령 이름",
        "lecture": "lecture01: document collection, docID, postings",
    },
    {
        "requirement": "2. 각 문서를 token으로 변환",
        "implementation": "영문 알파벳 정규식으로 token을 추출하고 소문자화합니다.",
        "evidence": "질의어 IDF 표와 결과 TF token",
        "lecture": "lecture02: tokenization",
    },
    {
        "requirement": "3. stemming",
        "implementation": "PorterStemmer로 문서와 질의에 같은 stemming pipeline을 적용합니다.",
        "evidence": "constitution -> constitut, people -> peopl 표시",
        "lecture": "lecture02: stemming",
    },
    {
        "requirement": "4. 사전 구성",
        "implementation": "전체 문서 term 집합을 dictionary로 구성합니다.",
        "evidence": "상단 Dictionary 5,492 terms",
        "lecture": "lecture01-02: dictionary",
    },
    {
        "requirement": "5. 평균 문서 길이와 IDF 계산",
        "implementation": "문서 길이 평균(avgdl)과 smoothing IDF를 계산합니다.",
        "evidence": "Avg Length 카드, 질의어 IDF 표",
        "lecture": "lecture06: df, idf, tf-idf",
    },
    {
        "requirement": "6. 역파일 생성",
        "implementation": "term -> {doc_id: tf} 형태의 inverted index를 생성합니다.",
        "evidence": "각 결과 카드의 질의 term별 TF",
        "lecture": "lecture01: inverted index, postings",
    },
    {
        "requirement": "7. cosine / BM25(k1=1.2, b=0.9) 순서화",
        "implementation": "TF-IDF cosine과 BM25 점수를 선택적으로 계산합니다.",
        "evidence": "Ranking method 라디오 버튼과 score pill",
        "lecture": "lecture06-07: vector space, cosine ranking",
    },
    {
        "requirement": "8. 중요 문장 추출",
        "implementation": "query term 빈도와 IDF 합이 큰 문장을 대표 snippet으로 선택합니다.",
        "evidence": "검색 결과의 인용문 형태 snippet",
        "lecture": "lecture08: dynamic summary",
    },
    {
        "requirement": "9. 결과 제공",
        "implementation": "대통령/연도, 관련 문장, score, TF, 원문 링크를 함께 제공합니다.",
        "evidence": "Ranked Addresses 카드와 취임사 원문 열기",
        "lecture": "lecture08: ranked results evaluation",
    },
]

LECTURE_MAP = [
    ("lecture01-intro", "Dictionary, postings, inverted index", "term별 postings와 docID 기반 결과 후보 생성"),
    ("lecture02-dictionary", "Tokenization, stemming, positional postings", "토큰화와 Porter stemming 적용, positional index는 과제 범위상 제외"),
    ("lecture03-tolerant", "Wildcard, k-gram, edit distance", "오타/와일드카드 검색은 미구현 항목으로 한계에 명시"),
    ("lecture04-construction", "Index construction", "작은 코퍼스를 메모리에서 한 번에 색인"),
    ("lecture05-compression", "Dictionary/postings compression", "59개 문서 규모라 압축은 생략, 보고서에서 설계 선택으로 설명"),
    ("lecture06-tfidf", "TF, DF, IDF, term weighting", "질의어 IDF와 문서별 TF를 화면에 노출"),
    ("lecture07-vectorspace", "Vector space, cosine similarity", "TF-IDF 벡터 정규화 후 cosine score 산출"),
    ("lecture08-evaluation", "Ranked results, dynamic summaries", "대표 문장 snippet과 수동 검증 질의로 결과 분석"),
]

ADDED_FEATURE = {
    "title": "보조자료 기반 추가 기능: Cosine vs BM25 랭킹 비교",
    "source": "Week 2 VectorDB/HNSW/Cosine 자료와 lecture06-07의 vector-space scoring을 연결",
    "value": "같은 질의가 cosine과 BM25에서 어떻게 다르게 순위화되는지 한 화면에서 비교해 랭킹 함수의 차이를 설명합니다.",
}


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
            margin: 14px 0 10px;
            max-width: 760px;
            font: 750 clamp(2.0rem, 4.2vw, 3.8rem)/0.96 "Fraunces", Georgia, serif;
            letter-spacing: -0.065em;
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

        .grading-hero {
            margin: 12px 0 18px;
            border: 1px solid rgba(23, 34, 28, 0.18);
            border-radius: 30px;
            padding: 22px;
            background:
                linear-gradient(135deg, rgba(23, 34, 28, 0.91), rgba(47, 81, 61, 0.82)),
                radial-gradient(circle at top right, rgba(180, 132, 50, 0.34), transparent 18rem);
            color: #fff7e8;
            box-shadow: var(--shadow);
        }

        .grading-hero .section-title,
        .grading-hero .hero-copy {
            color: #fff7e8;
        }

        .link-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .link-chip {
            display: inline-flex;
            align-items: center;
            padding: 9px 13px;
            border-radius: 999px;
            background: rgba(255, 247, 232, 0.12);
            border: 1px solid rgba(255, 247, 232, 0.28);
            color: #fff7e8 !important;
            font-weight: 800;
            text-decoration: none !important;
        }

        .grading-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 18px 0 24px;
        }

        .grading-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 16px;
            background: rgba(255, 252, 243, 0.76);
            box-shadow: 0 14px 34px rgba(42, 36, 24, 0.08);
        }

        .grading-card strong {
            display: block;
            margin-bottom: 7px;
            color: var(--moss);
            font: 750 1.02rem "Fraunces", Georgia, serif;
        }

        .mapping-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(255, 252, 243, 0.78);
            margin: 12px 0 22px;
        }

        .mapping-table th,
        .mapping-table td {
            padding: 12px 14px;
            border-bottom: 1px solid rgba(45, 55, 44, 0.12);
            vertical-align: top;
            text-align: left;
        }

        .mapping-table th {
            color: var(--moss);
            font-weight: 900;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(47, 81, 61, 0.08);
        }

        .mapping-table tr:last-child td { border-bottom: 0; }

        .rank-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin: 14px 0 26px;
        }

        .compare-list {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 16px;
            background: rgba(255, 252, 243, 0.76);
            box-shadow: 0 14px 34px rgba(42, 36, 24, 0.08);
        }

        .compare-list h4 {
            margin: 0 0 10px;
            font: 750 1.15rem "Fraunces", Georgia, serif;
            color: var(--moss);
        }

        .compare-row {
            display: grid;
            grid-template-columns: 42px 1fr auto;
            gap: 10px;
            align-items: center;
            padding: 9px 0;
            border-top: 1px solid rgba(45, 55, 44, 0.12);
        }

        .compare-row:first-of-type { border-top: 0; }
        .compare-rank {
            border-radius: 12px;
            background: rgba(23, 34, 28, 0.08);
            color: var(--ink);
            text-align: center;
            font-weight: 900;
            padding: 5px 0;
        }

        .compare-name { font-weight: 800; color: var(--ink); }
        .compare-score { color: #74511c; font-weight: 800; }

        @media (max-width: 860px) {
            .stat-grid, .checklist, .grading-grid, .rank-compare { grid-template-columns: 1fr; }
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


def render_grading_section() -> None:
    checklist_cards = "".join(
        f"""
        <div class="grading-card">
          <strong>{escape(item["requirement"])}</strong>
          <p>{escape(item["implementation"])}</p>
          <p><b>확인:</b> {escape(item["evidence"])}</p>
          <p><b>강의:</b> {escape(item["lecture"])}</p>
        </div>
        """
        for item in PROGRAM_CHECKS
    )
    lecture_rows = "".join(
        f"""
        <tr>
          <td>{escape(lecture)}</td>
          <td>{escape(concept)}</td>
          <td>{escape(application)}</td>
        </tr>
        """
        for lecture, concept, application in LECTURE_MAP
    )
    st.markdown(
        f"""
        <section class="grading-hero">
          <div class="eyebrow">Grading Evidence</div>
          <div class="section-title" style="font-size: 2rem;">채점 확인 요약</div>
          <p class="hero-copy">
            보고서 제출 전 교수/채점자가 요구사항, 소스, 데모, 강의 매핑을 빠르게 확인할 수 있도록
            구현 근거를 한 곳에 모았습니다. 이 탭은 Chrome/Safari의 인쇄 기능으로 PDF 저장하기 좋게 구성했습니다.
          </p>
          <div class="link-row">
            <a class="link-chip" href="{GITHUB_URL}">프로그램 소스: GitHub</a>
            <a class="link-chip" href="{DEMO_URL}">데모 사이트: {DEMO_URL}</a>
            <a class="link-chip" href="?q=constitution+people">검증 질의 예시</a>
          </div>
        </section>
        <div class="section-title">프로그램 개요 적용 체크리스트</div>
        <div class="grading-grid">{checklist_cards}</div>
        <div class="section-title">강의 내용 적용 매핑</div>
        <table class="mapping-table">
          <thead>
            <tr><th>강의 자료</th><th>수업 개념</th><th>프로젝트 적용 방식</th></tr>
          </thead>
          <tbody>{lecture_rows}</tbody>
        </table>
        <div class="section-title">보조자료 기반 추가 기능</div>
        <div class="grading-card" style="margin-bottom: 18px;">
          <strong>{escape(ADDED_FEATURE["title"])}</strong>
          <p><b>활용 자료:</b> {escape(ADDED_FEATURE["source"])}</p>
          <p><b>기능 의미:</b> {escape(ADDED_FEATURE["value"])}</p>
          <p><b>화면 확인:</b> 데모 탭의 Cosine vs BM25 랭킹 비교 패널에서 각 결과의 원문 링크까지 확인할 수 있습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rank_comparison(engine: InauguralSearchEngine, query: str) -> None:
    cosine_results = engine.search(query, method="cosine", top_k=5)
    bm25_results = engine.search(query, method="BM25", top_k=5)

    def render_list(title: str, results: list) -> str:
        rows = "".join(
            f"""
            <div class="compare-row">
              <span class="compare-rank">#{rank}</span>
              <span class="compare-name">
                {result.year} {escape(result.president)}
                <a class="doc-link" style="margin: 0 0 0 8px; font-size: 0.86rem;" href="{make_doc_url(result.file_id)}">원문</a>
              </span>
              <span class="compare-score">{result.score:.4f}</span>
            </div>
            """
            for rank, result in enumerate(results, start=1)
        )
        return f'<div class="compare-list"><h4>{title}</h4>{rows}</div>'

    st.markdown(
        f"""
        <div class="section-title">보조 기능: Cosine vs BM25 랭킹 비교</div>
        <p class="hero-copy" style="font-size: 0.98rem;">
          같은 질의를 두 랭킹 함수에 동시에 넣어 결과 순위 차이를 비교합니다.
          Vector-space cosine 관점과 BM25 문서 길이 보정 관점이 어떻게 다른지 확인하기 위한 기능입니다.
        </p>
        <div class="rank-compare">
          {render_list("Cosine TF-IDF Top 5", cosine_results)}
          {render_list("BM25 Top 5", bm25_results)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search_view(engine: InauguralSearchEngine) -> None:
    inject_design_system()
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">Information Retrieval Midterm</div>
          <div class="hero-title">IR Search Demo & Grading Evidence</div>
          <p class="hero-copy">
            NLTK 미국 대통령 취임사 검색 시스템입니다. 데모 탭에서는 기능을 직접 검증하고,
            채점/보고서 탭에서는 구현 근거와 강의 매핑을 확인합니다.
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
        """,
        unsafe_allow_html=True,
    )
    demo_tab, grading_tab = st.tabs(["데모", "채점/보고서"])

    with demo_tab:
        st.markdown(
            f"""
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

        if query.strip():
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

            render_rank_comparison(engine, query)
            render_results(engine, query, method, top_k)
        else:
            st.info("검색어를 입력하세요.")

    with grading_tab:
        render_grading_section()


def main() -> None:
    engine = get_engine()
    requested_file_id = st.query_params.get("doc")
    if requested_file_id:
        render_document_view(engine, requested_file_id)
        return
    render_search_view(engine)


if __name__ == "__main__":
    main()
