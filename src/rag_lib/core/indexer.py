from typing import List, Optional, Any, Dict
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from rag_lib.core.domain import Segment
from rag_lib.core.logger import logger

from rag_lib.processors.enricher import SegmentEnricher
# We avoid direct import of EntityExtractor for type hint if it causes circular import issues, 
# but here it seems fine as processors depends on core/graph, and core depends on processors?
# Actually processors depends on core (Segment). core (Indexer) depends on processors.
# This direction is fine: core -> processors -> core (Segment) is fine if imports are structured correctly.
from rag_lib.processors.entity_extractor import EntityExtractor

class Indexer:
    """
    Ingestion pipeline that takes Segments and pushes them to a VectorStore.
    Implements 'Parent-Child' style retrieval logic where we might embed the summary
    but store the full content in the payload.
    Also supports optional Graph extraction.
    """
    def __init__(
        self, 
        vector_store: VectorStore, 
        embeddings: Embeddings, 
        enricher: Optional[SegmentEnricher] = None,
        entity_extractor: Optional[EntityExtractor] = None
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.enricher = enricher
        self.entity_extractor = entity_extractor

    def index(self, segments: List[Segment], batch_size: int = 100) -> None:
        """
        Indexes a list of segments into the vector store.
        If entity_extractor is configured, also extracts graph data.
        """
        if not segments:
            logger.warning("Indexer received empty segments list.")
            return

        # 0. Enrich Segments (if configured)
        if self.enricher:
            logger.info("Enriching segments before indexing...")
            segments = self.enricher.enrich(segments)
            
        # 1. Graph Extraction (if configured)
        # We do this after enrichment so entities might benefit from nicer content, 
        # though currently enrichment adds metadata.
        if self.entity_extractor:
            logger.info("Extracting graph entities/relationships...")
            self.entity_extractor.process_segments(segments)

        logger.info(f"Indexing {len(segments)} segments with batch_size={batch_size}")

        # Prepare batches
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            self._process_batch(batch)
            logger.debug(f"Indexed batch {i // batch_size + 1}")

    def _process_batch(self, batch: List[Segment]) -> None:
        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for seg in batch:
            # 2. Decide what to embed (Vector Representative)
            # If we have a summary, use it for the vector (Semantic Key).
            # Otherwise use the raw content.
            summary = seg.metadata.get("summary")
            if summary:
                text_to_embed = summary
            else:
                text_to_embed = seg.content
            
            texts_to_embed.append(text_to_embed)
            
            # 3. Prepare Metadata (Payload)
            # We MUST store the original content so we can retrieve it.
            # We also flatten the existing metadata.
            payload = seg.metadata.copy()
            payload["content"] = seg.content
            payload["segment_id"] = seg.segment_id
            payload["type"] = seg.type.value
            payload["original_format"] = seg.original_format or "text"
            
            # If we embedded a summary, let's explicitly mark it
            if summary:
                payload["is_summary_embedding"] = True
            else:
                payload["is_summary_embedding"] = False

            metadatas.append(payload)
            ids.append(seg.segment_id)

        # 4. Add to VectorStore
        # Note: We rely on the VectorStore to handle the actual embedding generation
        # because langchain VectorStores (like Chroma) typically take an embedding function
        # in their constructor and 'add_texts' calls it.
        # However, if we didn't pass embeddings to the vector store constructor, this might fail.
        # But `rag_lib.vectors.factory` DOES pass embeddings to Chroma/Qdrant. 
        # So calling add_texts(texts) works.
        
        self.vector_store.add_texts(
            texts=texts_to_embed,
            metadatas=metadatas,
            ids=ids
        )

    async def aindex(self, segments: List[Segment], batch_size: int = 100) -> None:
        """
        Async version of index.
        """
        if not segments:
            return

        # 0. Enrich Segments (if configured)
        if self.enricher:
            logger.info("Enriching segments asynchronously before indexing...")
            segments = await self.enricher.aenrich(segments)
            
        # 1. Graph Extraction (if configured)
        if self.entity_extractor:
            logger.info("Extracting graph entities/relationships asynchronously...")
            await self.entity_extractor.aprocess_segments(segments)

        # Prepare batches
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            await self._process_batch_async(batch)

    async def _process_batch_async(self, batch: List[Segment]) -> None:
        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for seg in batch:
            summary = seg.metadata.get("summary")
            if summary:
                text_to_embed = summary
            else:
                text_to_embed = seg.content
            
            texts_to_embed.append(text_to_embed)
            
            payload = seg.metadata.copy()
            payload["content"] = seg.content
            payload["segment_id"] = seg.segment_id
            payload["type"] = seg.type.value
            payload["original_format"] = seg.original_format or "text"
            
            if summary:
                payload["is_summary_embedding"] = True
            else:
                payload["is_summary_embedding"] = False

            metadatas.append(payload)
            ids.append(seg.segment_id)

        # Most LangChain VectorStores support aadd_texts
        await self.vector_store.aadd_texts(
            texts=texts_to_embed,
            metadatas=metadatas,
            ids=ids
        )
