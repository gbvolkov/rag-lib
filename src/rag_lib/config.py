from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LLMSettings(BaseSettings):
    provider: str = "openai"
    model: str = "mini"
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

class GraphStoreSettings(BaseSettings):
    provider: str = "networkx"
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "neo4j"

    model_config = SettingsConfigDict(env_prefix="GRAPH_")

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
    table_summarizer_soft_max_chars: int = 700
    presentation_visual_summarizer_template: str = (
        "You analyze content extracted from a PowerPoint presentation.\n"
        "Summarize the visual in one concise factual paragraph.\n"
        "Visual kind: {visual_kind}\n"
        "Slide title: {slide_title}\n"
        "Shape name: {shape_name}\n"
        "Structured content:\n{structured_markdown}\n\n"
        "Summary:"
    )
    presentation_visual_summarizer_soft_max_chars: int = 500
    image_loader_summary_template: str = (
        "You analyze a document image.\n"
        "Use the image itself as the primary source and OCR text as supporting context.\n"
        "Image name: {image_name}\n"
        "OCR text:\n{ocr_text}\n\n"
        "Summary:"
    )
    image_loader_summary_soft_max_chars: int = 500

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
    graph_store: GraphStoreSettings = Field(default_factory=GraphStoreSettings)
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
