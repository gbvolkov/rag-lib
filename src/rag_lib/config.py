from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LLMSettings(BaseSettings):
    provider: str = "openai"
    model: str = "base"
    temperature: float = 0.0

    model_config = SettingsConfigDict(env_prefix="LLM_")

class EmbeddingsSettings(BaseSettings):
    provider: str = "openai"
    model_name: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

class VectorStoreSettings(BaseSettings):
    provider: str = "chroma"
    collection_name: str = "rag_lib_collection"
    path: str = "./chroma_db" # Default persistence directory
    url: Optional[str] = None
    api_key: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="VECTOR_")

class IngestionSettings(BaseSettings):
    chunk_size: int = 100 # CSV rows
    semantic_threshold: float = 0.6
    default_pdf_backend: str = "poppler"

    model_config = SettingsConfigDict(env_prefix="INGEST_")

class PromptSettings(BaseSettings):
    table_summarizer_template: str = (
        "You are a data analyst. I will provide a table in Markdown format.\n"
        "Your task is to summarize the content of the table in a concise, natural language paragraph.\n"
        "Highlight key entities, trends, or values. Do not just list the headers.\n\n"
        "Table:\n{table_content}\n\n"
        "Summary:"
    )

    model_config = SettingsConfigDict(env_prefix="PROMPT_")

class Settings(BaseSettings):
    """
    Centralized configuration for RAG Library.
    Reads from .env file or environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    
    # Global Settings
    log_level: str = "INFO"
    
    # Keep API Keys at top level for convenience/backward compat with env vars like OPENAI_API_KEY
    openai_api_key: Optional[str] = None
    openai_api_key_personal: Optional[str] = None
    mistral_api_key: Optional[str] = None
    ya_api_key: Optional[str] = None
    ya_folder_id: Optional[str] = None

# Global settings instance
settings = Settings()
