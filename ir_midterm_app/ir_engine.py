from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import nltk
from nltk.corpus import inaugural
from nltk.stem import PorterStemmer


LOCAL_NLTK_DATA = Path(__file__).resolve().parent / "nltk_data"
if LOCAL_NLTK_DATA.exists():
    nltk.data.path.append(str(LOCAL_NLTK_DATA))

TOKEN_RE = re.compile(r"[A-Za-z]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STEMMER = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """Return normalized tokens using the same pipeline for documents and queries."""
    # 중간고사 2-3번: 각 문서를 token으로 바꾸고 stemming을 수행한다.
    return [STEMMER.stem(match.group(0).lower()) for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: int
    file_id: str
    year: int
    president: str
    text: str
    tokens: list[str]
    term_freq: Counter[str]

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class SearchResult:
    doc_id: int
    file_id: str
    year: int
    president: str
    score: float
    snippet: str
    term_tf: dict[str, int]


class InauguralSearchEngine:
    """Small IR engine for the NLTK Inaugural Address corpus."""

    def __init__(self, documents: list[dict[str, object]]):
        if not documents:
            raise ValueError("documents must not be empty")

        # 중간고사 1번: doc id를 부여하고 doc id별 취임사/대통령 정보를 저장한다.
        self.documents = [self._to_indexed_document(document) for document in documents]
        self.document_count = len(self.documents)

        # 중간고사 4-6번: 사전, 문서 길이 통계, idf, 역파일을 만든다.
        self.dictionary: set[str] = set()
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        self.document_frequency: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.average_document_length = sum(doc.length for doc in self.documents) / self.document_count
        self._build_index()

    @classmethod
    def load_inaugural_documents(cls, stop_year: int = 2021) -> list[dict[str, object]]:
        """Load NLTK inaugural files through Biden by default."""
        try:
            file_ids = inaugural.fileids()
        except LookupError:
            nltk.download("inaugural", download_dir=str(LOCAL_NLTK_DATA))
            nltk.data.path.append(str(LOCAL_NLTK_DATA))
            file_ids = inaugural.fileids()

        documents: list[dict[str, object]] = []
        for doc_id, file_id in enumerate(file_ids):
            year, president = cls._parse_file_id(file_id)
            if year > stop_year:
                continue
            documents.append(
                {
                    "doc_id": len(documents),
                    "file_id": file_id,
                    "year": year,
                    "president": president,
                    "text": inaugural.raw(file_id),
                }
            )
        return documents

    @staticmethod
    def _parse_file_id(file_id: str) -> tuple[int, str]:
        year_text, president_text = file_id.removesuffix(".txt").split("-", maxsplit=1)
        return int(year_text), president_text

    @staticmethod
    def _to_indexed_document(document: dict[str, object]) -> IndexedDocument:
        text = str(document["text"])
        tokens = tokenize(text)
        return IndexedDocument(
            doc_id=int(document["doc_id"]),
            file_id=str(document["file_id"]),
            year=int(document["year"]),
            president=str(document["president"]),
            text=text,
            tokens=tokens,
            term_freq=Counter(tokens),
        )

    def _build_index(self) -> None:
        for document in self.documents:
            self.dictionary.update(document.term_freq)
            for term, tf in document.term_freq.items():
                self.inverted_index[term][document.doc_id] = tf

        self.document_frequency = {
            term: len(postings) for term, postings in self.inverted_index.items()
        }
        self.idf = {
            term: self._idf_from_df(df) for term, df in self.document_frequency.items()
        }

    def _idf_from_df(self, document_frequency: int) -> float:
        return math.log((self.document_count + 1) / (document_frequency + 1)) + 1

    def query_idf(self, query: str) -> dict[str, float]:
        """Return idf values for unique query terms."""
        return {term: self.idf.get(term, 0.0) for term in dict.fromkeys(tokenize(query))}

    def search(self, query: str, method: str = "cosine", top_k: int = 10) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        method_key = method.strip().lower()
        if method_key == "cosine":
            scores = self._cosine_scores(query_terms)
        elif method_key == "bm25":
            scores = self._bm25_scores(query_terms)
        else:
            raise ValueError("method must be either 'cosine' or 'BM25'")

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

        # 중간고사 8-9번: 찾아진 문서에서 중요 문장을 추출하고 결과로 제공한다.
        return [
            SearchResult(
                doc_id=doc_id,
                file_id=self.documents[doc_id].file_id,
                year=self.documents[doc_id].year,
                president=self.documents[doc_id].president,
                score=score,
                snippet=self.extract_snippet(self.documents[doc_id].text, query_terms),
                term_tf=self.term_tf_for_document(doc_id, query_terms),
            )
            for doc_id, score in ranked
        ]

    def _cosine_scores(self, query_terms: list[str]) -> dict[int, float]:
        # 중간고사 7번: cosine measure로 tf-idf 벡터를 순서화한다.
        query_tf = Counter(query_terms)
        query_weights = {
            term: tf * self.idf.get(term, self._idf_from_df(0))
            for term, tf in query_tf.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if query_norm == 0:
            return {}

        candidate_doc_ids = self._candidate_doc_ids(query_tf)
        scores: dict[int, float] = {}
        for doc_id in candidate_doc_ids:
            document = self.documents[doc_id]
            dot_product = 0.0
            doc_norm = 0.0
            for term, tf in document.term_freq.items():
                weight = tf * self.idf.get(term, 0.0)
                doc_norm += weight * weight
                dot_product += query_weights.get(term, 0.0) * weight
            if doc_norm > 0 and dot_product > 0:
                scores[doc_id] = dot_product / (math.sqrt(doc_norm) * query_norm)
            else:
                scores[doc_id] = 0.0
        return scores

    def _bm25_scores(self, query_terms: list[str], k1: float = 1.2, b: float = 0.9) -> dict[int, float]:
        # 중간고사 7번: BM25는 지정 파라미터 k1=1.2, b=0.9로 순서화한다.
        query_tf = Counter(query_terms)
        candidate_doc_ids = self._candidate_doc_ids(query_tf)
        scores: dict[int, float] = {}
        for doc_id in candidate_doc_ids:
            document = self.documents[doc_id]
            score = 0.0
            for term in query_tf:
                tf = document.term_freq.get(term, 0)
                if tf == 0:
                    continue
                length_factor = 1 - b + b * (document.length / self.average_document_length)
                numerator = tf * (k1 + 1)
                denominator = tf + (k1 * length_factor)
                score += self.idf.get(term, 0.0) * (numerator / denominator)
            scores[doc_id] = score
        return scores

    def _candidate_doc_ids(self, query_tf: Counter[str]) -> set[int]:
        candidate_doc_ids: set[int] = set()
        for term in query_tf:
            candidate_doc_ids.update(self.inverted_index.get(term, {}))
        return candidate_doc_ids or {document.doc_id for document in self.documents}

    def term_tf_for_document(self, doc_id: int, query_terms: list[str]) -> dict[str, int]:
        """Return per-query-term TF for one retrieved document."""
        document = self.documents[doc_id]
        return {term: document.term_freq.get(term, 0) for term in dict.fromkeys(query_terms)}

    def extract_snippet(self, text: str, query_terms: list[str]) -> str:
        """Pick the sentence with the largest sum of query-term idf values."""
        sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
        if not sentences:
            return text[:240]

        best_sentence = max(
            sentences,
            key=lambda sentence: self._sentence_score(sentence, query_terms),
        )
        return self._compact_snippet(best_sentence)

    def _sentence_score(self, sentence: str, query_terms: list[str]) -> float:
        sentence_tf = Counter(tokenize(sentence))
        return sum(sentence_tf.get(term, 0) * self.idf.get(term, 0.0) for term in query_terms)

    @staticmethod
    def _compact_snippet(sentence: str, max_length: int = 260) -> str:
        normalized = re.sub(r"\s+", " ", sentence).strip()
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max_length - 1].rstrip()}..."
