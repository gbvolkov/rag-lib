import os

from typing import Optional, Sequence, Any
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib  # built-in on Python 3.11+


from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None

try:
    from langchain_community.chat_models import ChatYandexGPT
except ImportError:
    ChatYandexGPT = None

from rag_lib.config import settings

VALID_MODES = {"base", "mini", "nano"}

@dataclass(frozen=True)
class ModelRegistry:
    providers: dict[str, dict[str, str]]

    @classmethod
    def from_toml(cls, path: Path) -> "ModelRegistry":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as f:
            data = tomllib.load(f)
        providers = data.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("Invalid config: missing top-level [providers] table.")

        # normalize to lowercase keys and validate modes
        norm: dict[str, dict[str, str]] = {}
        for prov, modes in providers.items():
            if not isinstance(modes, dict):
                raise ValueError(f"Invalid provider section for '{prov}'.")
            lower_modes = {k.lower(): v for k, v in modes.items()}
            # ensure exactly the three modes exist
            missing = VALID_MODES - set(lower_modes.keys())
            extra = set(lower_modes.keys()) - VALID_MODES
            if missing:
                raise ValueError(f"Provider '{prov}' missing modes: {sorted(missing)}.")
            if extra:
                raise ValueError(f"Provider '{prov}' has unknown modes: {sorted(extra)}.")
            norm[prov.lower()] = lower_modes
        return cls(norm)

    def get(self, provider: str, mode: str) -> str:
        p, m = provider.lower(), mode.lower()
        if p not in self.providers:
            known = ", ".join(sorted(self.providers.keys()))
            raise KeyError(f"Unknown provider '{provider}'. Known: {known}")
        if m not in VALID_MODES:
            raise KeyError(f"Unknown mode '{mode}'. Use one of: {', '.join(sorted(VALID_MODES))}")
        return self.providers[p][m]

@lru_cache(maxsize=1)
def _load_registry(config_path: str = "models.toml") -> ModelRegistry:
    return ModelRegistry.from_toml(Path(config_path))

def get_model(provider: str, mode: str, config_path: str = "models.toml") -> str:
    """
    Return the model string for a given provider and mode ('base' | 'mini' | 'nano').
    """
    # Include config_path in the cache key by passing it through _load_registry
    registry = _load_registry(config_path)
    return registry.get(provider, mode)

LLM_PROVIDER="openai"


def create_llm(
    model_name: Optional[str] = None,
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
    requested_model_name = model_name or settings.llm.model
    resolved_model_name = (
        get_model(provider, requested_model_name)
        if requested_model_name in VALID_MODES
        else requested_model_name
    )
    is_base_model = requested_model_name == "base"

    if temperature is None:
        temperature = settings.llm.temperature
    
    llm_model = resolved_model_name
    
    if provider == "openai":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Please install it.")
        # logic for 'base' verbosity
        verbosity = "low" if is_base_model else "medium"
        reasoning = "none" if is_base_model else "minimal"
        
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
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Please install it.")
        verbosity = "low" if is_base_model else "medium"
        reasoning = {
            "effort": "medium"
        } if is_base_model else {
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
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Please install it.")
        return ChatOpenAI(
            model=llm_model, 
            api_key=settings.openai_api_key,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "openai_pers":
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Please install it.")
        return ChatOpenAI(
            model=llm_model, 
            api_key=settings.openai_api_key_personal,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "mistral":
        if ChatMistralAI is None:
            raise ImportError("langchain-mistralai is not installed. Please install it.")
        return ChatMistralAI(
            model=llm_model, 
            api_key=settings.mistral_api_key,
            temperature=temperature, 
            frequency_penalty=frequency_penalty,
            streaming=streaming,
            callbacks=list(callbacks) if callbacks else None,
        )

    elif provider == "yandex":
        if ChatYandexGPT is None:
            raise ImportError("langchain-community is not installed or ChatYandexGPT is missing.")
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
