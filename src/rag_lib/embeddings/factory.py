import os
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from rag_lib.config import settings

def get_embeddings_model(
    provider: Optional[str] = None, 
    model_name: Optional[str] = None
) -> Embeddings:
    """
    Factory to get Embeddings model.
    """
    provider = provider or settings.embeddings.provider
    model_name = model_name or settings.embeddings.model_name

    if provider == "openai":
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            api_key=settings.openai_api_key
        )
    
    elif provider == "local" or provider == "huggingface":
        # Default to the user-requested multilingual model
        model = model_name or "intfloat/multilingual-e5-large"
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'}, # Can be configured to cuda
            encode_kwargs={'normalize_embeddings': True}
        )
        
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")
