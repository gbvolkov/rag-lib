from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from rag_lib.core.domain import Segment
from rag_lib.raptor.clustering import ClusteringService
from rag_lib.raptor.summarization import ClusterSummarizer
from rag_lib.raptor.tree_builder import TreeBuilder
from rag_lib.core.logger import logger

class RaptorProcessor:
    """
    Processor that implements RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).
    It takes a list of Segments (leaves) and returns an expanded list including hierarchical summaries.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        embeddings: Embeddings,
        max_levels: int = 3,
        clustering_service: Optional[ClusteringService] = None,
        summary_prompt_template: str | None = None,
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.max_levels = max_levels
        
        # Initialize components
        self.clustering = clustering_service or ClusteringService()
        self.summarizer = ClusterSummarizer(
            llm=llm,
            summary_prompt_template=summary_prompt_template,
        )
        self.builder = TreeBuilder(
            clustering_service=self.clustering,
            summarizer=self.summarizer,
            embeddings_model=self.embeddings
        )

    def process_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Synchronous processing.
        """
        logger.info("Starting RAPTOR processing...")
        return self.builder.build(segments, n_levels=self.max_levels)

    async def aprocess_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Asynchronous processing.
        """
        logger.info("Starting RAPTOR processing (Async)...")
        return await self.builder.abuild(segments, n_levels=self.max_levels)
