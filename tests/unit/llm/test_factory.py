import pytest
from unittest.mock import patch, MagicMock
from rag_lib.llm.factory import get_llm

def test_get_openai_base():
    with patch("rag_lib.llm.factory.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        
        get_llm(model="base", provider="openai")
        
        mock_openai.assert_called_with(
            model="gpt-4o-mini",
            model_kwargs={
                "max_tool_calls": 3,
                "reasoning": {"effort": "none"},
                "verbosity": "low",
                "use_previous_response_id": True
            },
            temperature=0,
            frequency_penalty=None,
            streaming=True,
            callbacks=None
        )

def test_get_openai_think():
    with patch("rag_lib.llm.factory.ChatOpenAI") as mock_openai:
        # We need to patch os.getenv too? Or just assume it works.
        # Patching getenv for API key
        with patch("os.getenv", return_value="sk-test"):
            get_llm(model="base", provider="openai_think")
            
            # Check reasoning is medium
            call_args = mock_openai.call_args
            assert call_args.kwargs['model_kwargs']['reasoning'] == {'effort': 'medium'}
            assert call_args.kwargs['api_key'] == "sk-test"

def test_get_mistral():
    with patch("rag_lib.llm.factory.ChatMistralAI") as mock_mistral:
        get_llm(model="mistral-large", provider="mistral")
        mock_mistral.assert_called()
        assert mock_mistral.call_args.kwargs['model'] == "mistral-large"

def test_get_yandex():
    # Attempt to import ChatYandexGPT mock
    # Since we use a try-import block, we must patch where it's imported or the factory itself
    # If the import failed in factory, ChatYandexGPT is Any (not callable easily without more mocks)
    # But for now let's assume we can patch `rag_lib.llm.factory.ChatYandexGPT`
    
    with patch("rag_lib.llm.factory.ChatYandexGPT") as mock_yandex:
        with patch("os.getenv") as mock_getenv:
            def side_effect(key):
                if key == "YA_FOLDER_ID": return "folder-123"
                if key == "YA_API_KEY": return "api-key-123"
                return None
            mock_getenv.side_effect = side_effect
            
            get_llm(model="yandexgpt-lite", provider="yandex")
            
            mock_yandex.assert_called_with(
                api_key="api-key-123",
                folder_id="folder-123",
                model_uri="gpt://folder-123/yandexgpt-lite",
                temperature=0
            )
