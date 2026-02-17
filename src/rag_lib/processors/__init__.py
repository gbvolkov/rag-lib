from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.processors.community_summarizer import CommunitySummarizer

__all__ = [
    "SegmentEnricher",
    "EntityExtractor",
    "CommunitySummarizer",
    "RaptorProcessor",
]


def __getattr__(name: str):
    if name == "RaptorProcessor":
        from rag_lib.processors.raptor import RaptorProcessor

        return RaptorProcessor
    raise AttributeError(f"module 'rag_lib.processors' has no attribute '{name}'")
