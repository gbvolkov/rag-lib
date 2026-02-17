import pytest
from unittest.mock import patch, MagicMock
from rag_lib.vectors.factory import get_vector_store

# Mock Embeddings
mock_embeddings = MagicMock()

def test_get_chroma():
    with patch("rag_lib.vectors.factory.Chroma") as mock_chroma:
        get_vector_store(provider="chroma", embeddings=mock_embeddings)
        mock_chroma.assert_called_with(
            collection_name="rag_collection",
            embedding_function=mock_embeddings,
            persist_directory="./chroma_db"
        )

def test_get_qdrant_memory():
    with patch("rag_lib.vectors.factory.Qdrant") as mock_qdrant:
        get_vector_store(provider="qdrant", embeddings=mock_embeddings)
        mock_qdrant.assert_called_with(
            client=None,
            collection_name="rag_collection",
            embeddings=mock_embeddings,
            location=":memory:"
        )

def test_get_qdrant_server():
    with patch("rag_lib.vectors.factory.Qdrant") as mock_qdrant:
        # For server, it calls from_existing_collection
        get_vector_store(
            provider="qdrant", 
            embeddings=mock_embeddings, 
            connection_string="http://localhost:6333"
        )
        mock_qdrant.from_existing_collection.assert_called_with(
            embedding=mock_embeddings,
            collection_name="rag_collection",
            url="http://localhost:6333"
        )

def test_get_postgres():
    with patch("rag_lib.vectors.factory.PGVector") as mock_pg:
        conn = "postgresql+psycopg2://user:pass@localhost:5432/db"
        get_vector_store(
            provider="postgres", 
            embeddings=mock_embeddings, 
            connection_string=conn
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
        store = get_vector_store(provider="faiss", embeddings=mock_embeddings)
        mock_faiss.from_texts.assert_called_with([""], mock_embeddings)
        assert store == "MockFaissStore"

def test_missing_embeddings():
    with pytest.raises(ValueError):
        get_vector_store(provider="chroma", embeddings=None)
