import pytest
from unittest.mock import patch, MagicMock
from rag_lib.vectors.factory import create_vector_store, _strict_chroma_cosine_relevance

# Mock Embeddings
mock_embeddings = MagicMock()

def test_create_chroma():
    with patch("rag_lib.vectors.factory.Chroma") as mock_chroma:
        create_vector_store(provider="chroma", embeddings=mock_embeddings)
        kwargs = mock_chroma.call_args.kwargs
        assert kwargs["collection_name"] == "rag_collection"
        assert kwargs["embedding_function"] is mock_embeddings
        assert kwargs["persist_directory"] == "./chroma_db"
        assert kwargs["collection_configuration"] == {"hnsw": {"space": "cosine"}}
        assert callable(kwargs["relevance_score_fn"])

def test_create_qdrant_memory():
    with patch("rag_lib.vectors.factory.Qdrant") as mock_qdrant:
        create_vector_store(provider="qdrant", embeddings=mock_embeddings)
        mock_qdrant.assert_called_with(
            client=None,
            collection_name="rag_collection",
            embeddings=mock_embeddings,
            location=":memory:"
        )

def test_create_qdrant_server_with_canonical_connection_uri():
    with patch("rag_lib.vectors.factory.Qdrant") as mock_qdrant:
        # For server, it calls from_existing_collection
        create_vector_store(
            provider="qdrant", 
            embeddings=mock_embeddings, 
            connection_uri="http://localhost:6333"
        )
        mock_qdrant.from_existing_collection.assert_called_with(
            embedding=mock_embeddings,
            collection_name="rag_collection",
            url="http://localhost:6333"
        )

def test_create_postgres():
    with patch("rag_lib.vectors.factory.PGVector") as mock_pg:
        conn = "postgresql+psycopg2://user:pass@localhost:5432/db"
        create_vector_store(
            provider="postgres", 
            embeddings=mock_embeddings, 
            connection_uri=conn
        )
        mock_pg.assert_called_with(
            embeddings=mock_embeddings,
            collection_name="rag_collection",
            connection=conn,
            use_jsonb=True
        )

def test_faiss_creation():
    with patch("rag_lib.vectors.factory.FAISS") as mock_faiss:
        mock_faiss.from_texts.return_value = "MockFaissStore"
        store = create_vector_store(provider="faiss", embeddings=mock_embeddings)
        mock_faiss.from_texts.assert_called_with([""], mock_embeddings)
        assert store == "MockFaissStore"

def test_missing_embeddings():
    with pytest.raises(ValueError):
        create_vector_store(provider="chroma", embeddings=None)


def test_strict_chroma_cosine_relevance_mapping():
    assert _strict_chroma_cosine_relevance(0.0) == pytest.approx(1.0)
    assert _strict_chroma_cosine_relevance(1.0) == pytest.approx(0.5)
    assert _strict_chroma_cosine_relevance(2.0) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        _strict_chroma_cosine_relevance(-0.01)
    with pytest.raises(ValueError):
        _strict_chroma_cosine_relevance(2.01)
