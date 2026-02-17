import pytest
from unittest.mock import patch, MagicMock
from rag_lib.embeddings import factory as embeddings_factory
from rag_lib.embeddings.factory import get_embeddings_model

def test_get_openai_embeddings():
    with patch.object(embeddings_factory.settings, "openai_api_key", "test-openai-key"):
        with patch("rag_lib.embeddings.factory.OpenAIEmbeddings") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            # Test Default
            result = get_embeddings_model(provider="openai")
            mock_openai.assert_called_with(model="text-embedding-3-small", api_key="test-openai-key")
            assert result == mock_instance

def test_get_local_embeddings():
    mock_instance = MagicMock()
    with patch("rag_lib.embeddings.factory.HuggingFaceEmbeddings") as mock_hf:
        mock_hf.return_value = mock_instance

        # Test Default
        result = get_embeddings_model(provider="local")
        mock_hf.assert_called_with(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        assert result == mock_instance

def test_invalid_provider():
    with pytest.raises(ValueError):
        get_embeddings_model(provider="unknown")
