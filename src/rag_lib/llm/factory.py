import os
from typing import Optional, Sequence, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
# Assuming generic community implementation or wrapper for Yandex if specific lib not found
# The user snippet used config.YA_FOLDER_ID, etc. We'll use os.getenv/explicit args.
try:
    from langchain_community.chat_models import ChatYandexGPT
except ImportError:
    # Fallback or placeholder if dependency issues
    ChatYandexGPT = Any

from rag_lib.config import settings

def get_llm(
    model: Optional[str] = None, 
    provider: Optional[str] = None,
    temperature: Optional[float] = None, 
    frequency_penalty: Optional[float] = None,
    *,
    streaming: bool = True,
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,    
) -> BaseChatModel:
    """
    Factory to get LLM Chat Model based on provider and configuration.
    Reflects specific user requirements for 'openai', 'openai_think', 'yandex', etc.
    """
    # Load defaults from settings if not provided
    provider = provider or settings.llm.provider
    model = model or settings.llm.model
    if temperature is None:
        temperature = settings.llm.temperature
    
    # Determine Model Name based on aliases or pass through
    # Real implementation would map "base" -> "gpt-4o-mini" etc.
    # For now we use exact strings or placeholders
    llm_model = model 
    if provider == "openai" and model == "base":
        llm_model = "gpt-4o-mini" # Example default
    
    if provider == "openai":
        # logic for 'base' verbosity
        verbosity = "low" if model == "base" else "medium"
        reasoning = "none" if model == "base" else "minimal"
        
        return ChatOpenAI(
            model=llm_model,
            api_key=settings.openai_api_key, # explicit from settings
            model_kwargs={
                "max_tool_calls": 3,
                "use_previous_response_id": True
            },
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "openai_think":
        verbosity = "low" if model == "base" else "medium"
        reasoning = {
            "effort": "medium"
        } if model == "base" else {
            "effort": "minimal"
        }

        return ChatOpenAI(
            model=llm_model,
            api_key=settings.openai_api_key_personal, # Specific key
            model_kwargs={
                "max_tool_calls": 3,
                "reasoning": reasoning,
                "verbosity": verbosity,
                "use_previous_response_id": True
            },
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "openai_4":
        return ChatOpenAI(
            model=llm_model, 
            api_key=settings.openai_api_key,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "openai_pers":
        return ChatOpenAI(
            model=llm_model, 
            api_key=settings.openai_api_key_personal,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "mistral":
        return ChatMistralAI(
            model=llm_model, 
            api_key=settings.mistral_api_key,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "yandex":
        # constructing model uri
        folder_id = settings.ya_folder_id
        api_key = settings.ya_api_key
        model_uri = f'gpt://{folder_id}/{llm_model}'
        return ChatYandexGPT(
            api_key=api_key,
            folder_id=folder_id,
            model_uri=model_uri,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
