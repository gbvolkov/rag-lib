from rag_lib.config import Settings

def test_default_settings():
    settings = Settings()
    assert settings.llm.provider == "openai"
    assert settings.vector_store.provider == "chroma"
    assert settings.ingestion.semantic_threshold == 0.6

def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    monkeypatch.setenv("LLM_MODEL", "mistral-large")
    monkeypatch.setenv("INGEST_SEMANTIC_THRESHOLD", "0.8")

    settings = Settings()
    assert settings.llm.provider == "mistral"
    assert settings.llm.model == "mistral-large"
    assert settings.ingestion.semantic_threshold == 0.8

def test_api_keys_optional():
    settings = Settings()
    assert settings.openai_api_key is None or isinstance(settings.openai_api_key, str)
