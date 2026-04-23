import math

from ir_engine import InauguralSearchEngine, tokenize


def test_tokenize_lowercases_filters_non_letters_and_stems():
    assert tokenize("Freedom, freedoms! 1776 UNION.") == ["freedom", "freedom", "union"]


def test_engine_builds_dictionary_inverted_index_tf_and_idf():
    docs = [
        {
            "doc_id": 0,
            "file_id": "1789-Washington.txt",
            "year": 1789,
            "president": "Washington",
            "text": "Freedom and union. Freedom for all.",
        },
        {
            "doc_id": 1,
            "file_id": "1797-Adams.txt",
            "year": 1797,
            "president": "Adams",
            "text": "Union and peace.",
        },
    ]

    engine = InauguralSearchEngine(docs)

    assert "freedom" in engine.dictionary
    assert engine.inverted_index["freedom"][0] == 2
    assert engine.inverted_index["union"] == {0: 1, 1: 1}
    assert engine.documents[0].term_freq["freedom"] == 2
    assert math.isclose(engine.idf["freedom"], math.log((2 + 1) / (1 + 1)) + 1)
    assert engine.average_document_length > 0


def test_cosine_and_bm25_return_ranked_results_with_explanations():
    docs = [
        {
            "doc_id": 0,
            "file_id": "1789-Washington.txt",
            "year": 1789,
            "president": "Washington",
            "text": "Freedom and union. Freedom for all citizens.",
        },
        {
            "doc_id": 1,
            "file_id": "1797-Adams.txt",
            "year": 1797,
            "president": "Adams",
            "text": "Union and friendship with all nations.",
        },
    ]
    engine = InauguralSearchEngine(docs)

    cosine_results = engine.search("freedom union", method="cosine", top_k=2)
    bm25_results = engine.search("freedom union", method="BM25", top_k=2)

    assert cosine_results[0].president == "Washington"
    assert bm25_results[0].president == "Washington"
    assert cosine_results[0].score > cosine_results[1].score
    assert "Freedom and union" in cosine_results[0].snippet
    assert cosine_results[0].term_tf == {"freedom": 2, "union": 1}
    assert engine.query_idf("freedom union")["freedom"] == engine.idf["freedom"]


def test_default_corpus_loader_stops_at_biden():
    docs = InauguralSearchEngine.load_inaugural_documents()

    assert docs[-1]["file_id"] == "2021-Biden.txt"
    assert all(doc["year"] <= 2021 for doc in docs)
