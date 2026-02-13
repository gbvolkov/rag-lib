import pytest
import os
from unittest.mock import MagicMock, patch
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.loaders.csv_excel import CSVLoader
from rag_lib.summarizers.table_llm import LLMTableSummarizer
from rag_lib.config import Settings, settings

@pytest.fixture
def clean_env():
    # Save original env
    old_env = os.environ.copy()
    yield
    # Restore
    os.environ.clear()
    os.environ.update(old_env)

def test_chunker_default_from_config(clean_env):
    # Default is 0.6
    # Note: SemanticChunker instantiates Settings() if threshold is None.
    # But Settings() reads env vars at instantiation.
    
    # 1. Verify default
    chunker = SemanticChunker(embeddings=MagicMock())
    assert chunker.threshold == 0.6
    
    # 2. Change env
    os.environ["INGEST_SEMANTIC_THRESHOLD"] = "0.85"
    
    # We must ensure Settings re-reads env or we instantiate new Settings
    # SemanticChunker does `Settings().ingestion.semantic_threshold`
    chunker_env = SemanticChunker(embeddings=MagicMock())
    assert chunker_env.threshold == 0.85

def test_csv_loader_default_from_config(clean_env):
    # Default is 100
    loader = CSVLoader("dummy.csv")
    assert loader.chunk_size == 100
    
    # Change env
    os.environ["INGEST_CHUNK_SIZE"] = "500"
    
    loader_env = CSVLoader("dummy.csv")
    assert loader_env.chunk_size == 500

def test_summarizer_prompt_from_config(clean_env):
    llm = MagicMock()
    
    # Default prompt checks
    summarizer = LLMTableSummarizer(llm)
    # The prompt object in langchain is complex
    template_str = summarizer.prompt.messages[0].prompt.template
    assert "You are a data analyst" in template_str
    
    # Change env
    custom_prompt = "You are a Medical Doctor. Summarize this patient data."
    os.environ["PROMPT_TABLE_SUMMARIZER_TEMPLATE"] = custom_prompt
    
    # Re-instantiate
    summarizer_env = LLMTableSummarizer(llm)
    template_str_env = summarizer_env.prompt.messages[0].prompt.template
    assert "Medical Doctor" in template_str_env
