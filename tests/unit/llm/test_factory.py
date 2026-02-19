import pytest
from unittest.mock import patch, MagicMock
from rag_lib.llm import factory as llm_factory
from rag_lib.llm.factory import create_llm

def test_get_openai_base():
    with patch.object(llm_factory.settings, "openai_api_key", "sk-openai"):
        with patch("rag_lib.llm.factory.ChatOpenAI") as mock_openai:
            mock_instance = MagicMock()
            mock_openai.return_value = mock_instance
            
            create_llm(model_name="base", provider="openai", temperature=0.0)
            
            mock_openai.assert_called_with(
                model="gpt-4o-mini",
                api_key="sk-openai",
                model_kwargs={
                    "max_tool_calls": 3,
                    "use_previous_response_id": True,
                },
                temperature=0.0,
                frequency_penalty=None,
                streaming=True,
                callbacks=None,
            )

def test_get_openai_think():
    with patch.object(llm_factory.settings, "openai_api_key_personal", "sk-think"):
        with patch("rag_lib.llm.factory.ChatOpenAI") as mock_openai:
            create_llm(model_name="base", provider="openai_think", temperature=0.0)
            
            call_args = mock_openai.call_args
            assert call_args.kwargs["model_kwargs"]["reasoning"] == {"effort": "medium"}
            assert call_args.kwargs["api_key"] == "sk-think"

def test_get_mistral():
    with patch.object(llm_factory.settings, "mistral_api_key", "mistral-key"):
        with patch("rag_lib.llm.factory.ChatMistralAI") as mock_mistral:
            create_llm(model_name="mistral-large", provider="mistral", temperature=0.0)
            mock_mistral.assert_called()
            assert mock_mistral.call_args.kwargs["model"] == "mistral-large"
            assert mock_mistral.call_args.kwargs["api_key"] == "mistral-key"

def test_get_yandex():
    with patch.object(llm_factory.settings, "ya_folder_id", "folder-123"):
        with patch.object(llm_factory.settings, "ya_api_key", "api-key-123"):
            with patch("rag_lib.llm.factory.ChatYandexGPT") as mock_yandex:
                create_llm(model_name="yandexgpt-lite", provider="yandex", temperature=0.0)
                
                mock_yandex.assert_called_with(
                    api_key="api-key-123",
                    folder_id="folder-123",
                    model_uri="gpt://folder-123/yandexgpt-lite",
                    temperature=0.0,
                )

