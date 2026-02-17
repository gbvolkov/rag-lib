from typing import List, Optional, Any, Dict
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
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
    Also supports optional Graph extraction and Dual Storage (DocStore).
    """
    def __init__(
        self, 
        vector_store: VectorStore, 
        embeddings: Embeddings, 
        enricher: Optional[SegmentEnricher] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        doc_store: Optional[BaseStore[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.enricher = enricher
        self.entity_extractor = entity_extractor
        self.doc_store = doc_store

    def index(self, segments: List[Segment], parents: Optional[List[Segment]] = None, batch_size: int = 100) -> None:
        """
        Indexes a list of segments into the vector store.
        
        Args:
            segments: List of segments (or chunks) to be indexed in VectorStore.
            parents: Optional list of parent segments to be stored in DocStore (Dual Storage).
                     If provided, these are stored in doc_store (if configured).
                     If NOT provided but doc_store IS configured, 'segments' are stored in doc_store.
            batch_size: Batch size for vector indexing.
        """
        if not segments:
            logger.warning("Indexer received empty segments list.")
            return

        # 0. Enrich Segments (Vector/Chunks) (if configured)
        # Note: We typically enrich the chunks (segments) before embedding.
        if self.enricher:
            logger.info("Enriching segments before indexing...")
            segments = self.enricher.enrich(segments)
            
        # 1. Graph Extraction (if configured)
        if self.entity_extractor:
            logger.info("Extracting graph entities/relationships...")
            self.entity_extractor.process_segments(segments)

        # 2. DocStore Storage (Dual Storage Pattern)
        if self.doc_store:
            docs_to_store = parents if parents else segments
            logger.info(f"Storing {len(docs_to_store)} documents in DocStore...")
            
            # Map segment_id -> LangChain Document (for compatibility with Retrievers)
            # Retrievers like MultiVectorRetriever expect values to be Documents.
            pairs = [(s.segment_id, s.to_langchain()) for s in docs_to_store]
            try:
                self.doc_store.mset(pairs)
                logger.info("Successfully populated DocStore.")
            except Exception as e:
                logger.error(f"Failed to store documents in DocStore: {e}")
                # We log but continue? Or raise? Use strictness: Raise.
                raise e

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
            # If using Dual Storage (DocStore), the VectorStore payload can be minimal (just ID + basic meta).
            # But we keep full metadata for filtering.
            payload = seg.metadata.copy()
            # We strictly ensure 'content' is present in metadata for Self-Retrieval if no DocStore,
            # but if using DocStore, it is redundant but harmless.
            payload["content"] = seg.content 
            payload["segment_id"] = seg.segment_id
            if seg.parent_id:
                payload["parent_id"] = seg.parent_id
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
        self.vector_store.add_texts(
            texts=texts_to_embed,
            metadatas=[self._flatten_metadata(m) for m in metadatas],
            ids=ids
        )

    async def aindex(self, segments: List[Segment], parents: Optional[List[Segment]] = None, batch_size: int = 100) -> None:
        """
        Async version of index.
        """
        if not segments:
            return

        # 0. Enrich Segments
        if self.enricher:
            logger.info("Enriching segments asynchronously before indexing...")
            segments = await self.enricher.aenrich(segments)
            
        # 1. Graph Extraction
        if self.entity_extractor:
            logger.info("Extracting graph entities/relationships asynchronously...")
            await self.entity_extractor.aprocess_segments(segments)

        # 2. DocStore Storage
        if self.doc_store:
            docs_to_store = parents if parents else segments
            logger.info(f"Storing {len(docs_to_store)} documents in DocStore (Async)...")
            pairs = [(s.segment_id, s.to_langchain()) for s in docs_to_store]
            try:
                await self.doc_store.amset(pairs)
            except Exception as e:
                 logger.error(f"Failed to async store documents: {e}")
                 raise e

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
            if seg.parent_id:
                payload["parent_id"] = seg.parent_id
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
            metadatas=[self._flatten_metadata(m) for m in metadatas],
            ids=ids
        )

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flattens complex metadata values (lists, dicts, objects) to strings for VectorStore compatibility.
        """
        flat = {}
        for k, v in metadata.items():
            if v is None:
                flat[k] = "" # None causes issues in some stores
                continue
                
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif isinstance(v, (list, tuple)):
                flat[k] = ", ".join(map(str, v))
            elif isinstance(v, dict):
                import json
                try:
                    flat[k] = json.dumps(v)
                except:
                    flat[k] = str(v)
            else:
                # Catch-all for objects like Path, etc.
                flat[k] = str(v)
        return flat
