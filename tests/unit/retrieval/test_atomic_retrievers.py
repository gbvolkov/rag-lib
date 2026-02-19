import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock
from rag_lib.retrieval.retrievers import (
    RegexRetriever, 
    FuzzyRetriever, 
    create_vector_retriever, 
    create_bm25_retriever,
)

# Test Data
DOCS = [
    Document(page_content="The ID is 12345.", metadata={"id": "doc1"}),
    Document(page_content="Refer to code ABC-99.", metadata={"id": "doc2"}),
    Document(page_content="Apple Pie Recipe", metadata={"type": "food"}),
    Document(page_content="Appel Pie Recip", metadata={"type": "typo"}) # For fuzzy
]

def test_regex_retriever():
    retriever = RegexRetriever(documents=DOCS)
    
    # Exact match
    results = retriever.invoke("12345")
    assert len(results) == 1
    assert results[0].metadata["id"] == "doc1"
    
    # Partial match
    results = retriever.invoke("ABC-99")
    assert len(results) == 1
    assert results[0].metadata["id"] == "doc2"
    
    # No match
    results = retriever.invoke("XYZ-00")
    assert len(results) == 0

def test_fuzzy_retriever():
    try:
        import rapidfuzz
    except ImportError:
        pytest.skip("rapidfuzz not installed")
        
    retriever = FuzzyRetriever(documents=DOCS, threshold=70)
    
    # Typo match "Apple" vs "Appel"
    results = retriever.invoke("Appel Pie") 
    # Should match both "Apple Pie Recipe" (similar) and "Appel Pie Recip" (exactish)
    assert len(results) >= 1
    ids = [d.metadata.get("type") for d in results]
    assert "food" in ids or "typo" in ids

def test_vector_retriever_factory():
    mock_store = MagicMock()
    mock_store.as_retriever.return_value = "MockRetriever"
    
    retriever = create_vector_retriever(mock_store, top_k=10)
    assert retriever == "MockRetriever"
    mock_store.as_retriever.assert_called_with(
        search_type="similarity",
        search_kwargs={"k": 10}
    )


def test_bm25_retriever_factory():
    # BM25 requires rank_bm25 package
    try:
        import rank_bm25
    except ImportError:
        pytest.skip("rank_bm25 not installed")
        
    retriever = create_bm25_retriever(DOCS, top_k=2)
    results = retriever.invoke("Apple")
    assert len(results) > 0
    assert "Apple" in results[0].page_content or "Appel" in results[0].page_content

