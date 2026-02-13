import os
import pytest
from rag_lib.config import Settings

def test_default_settings():
    # Ensure clean env
    # Pydantic settings reads env vars, so we must be careful not to pick up real ones if set
    # But for defaults, assuming env is empty of these specific keys
    settings = Settings()
    assert settings.LLM_PROVIDER == "openai"
    assert settings.VECTOR_STORE_PROVIDER == "chroma"
    assert settings.SEMANTIC_CHUNK_THRESHOLD == 0.6

def test_env_override():
    os.environ["LLM_PROVIDER"] = "mistral"
    os.environ["LLM_MODEL"] = "mistral-large"
    os.environ["SEMANTIC_CHUNK_THRESHOLD"] = "0.8"
    
    try:
        # Pydantic caches, so we might need to reload or reinstantiate
        settings = Settings()
        assert settings.LLM_PROVIDER == "mistral"
        assert settings.LLM_MODEL == "mistral-large"
        assert settings.SEMANTIC_CHUNK_THRESHOLD == 0.8
    finally:
        # Cleanup
        del os.environ["LLM_PROVIDER"]
        del os.environ["LLM_MODEL"]
        del os.environ["SEMANTIC_CHUNK_THRESHOLD"]

def test_api_keys_optional():
    settings = Settings()
    assert settings.OPENAI_API_KEY is None or isinstance(settings.OPENAI_API_KEY, str)
