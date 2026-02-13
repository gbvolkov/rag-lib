from typing import List, Optional, Any, Dict
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from rag_lib.core.domain import Segment
from rag_lib.core.logger import logger

class IndexBuilder:
    """
    Orchestrates the indexing process by separating Index (Vector) from Storage (Document/Segment).
    Pattern:
    1. Segments -> DocumentStore (Full Fidelity)
    2. Segments -> VectorStore (Embeddings + Link to DocumentStore)
    """
    def __init__(self, vector_store: VectorStore, doc_store: BaseStore[str, Segment]):
        self.vector_store = vector_store
        self.doc_store = doc_store

    def build(self, segments: List[Segment], batch_size: int = 100) -> None:
        """
        Synchronous build execution.
        """
        if not segments:
            logger.warning("IndexBuilder received empty segments list.")
            return

        logger.info(f"Building index for {len(segments)} segments...")

        # 1. Store Segments in DocumentStore (Bulk)
        # We assume doc_store can handle bulk.
        pairs = [(s.segment_id, s) for s in segments]
        try:
            self.doc_store.mset(pairs)
            logger.info(f"Persisted {len(segments)} segments to DocumentStore.")
        except Exception as e:
            logger.error(f"Failed to persist segments to DocumentStore: {e}")
            raise e

        # 2. Add to VectorStore (Batched)
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            self._process_vector_batch(batch)
            logger.debug(f"Indexed vector batch {i // batch_size + 1}")

    def _process_vector_batch(self, batch: List[Segment]) -> None:
        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for seg in batch:
            # Logic: Embed Summary if present, else Content
            summary = seg.metadata.get("summary")
            if summary:
                text_to_embed = summary
                is_summary = True
            else:
                text_to_embed = seg.content
                is_summary = False
            
            texts_to_embed.append(text_to_embed)
            
            # Payload Construction
            # We include 'segment_id' as the Foreign Key to DocumentStore.
            # We also include standard metadata for filtering.
            # We DO NOT strictly need to include 'content' here if we rely on DocStore for retrieval,
            # BUT keeping it small/metadata-only is key for "Dual Storage".
            # However, for convenience in debug or hybrid search, we keep metadata.
            
            payload = seg.metadata.copy()
            payload["segment_id"] = seg.segment_id
            payload["type"] = seg.type.value
            payload["is_summary_embedding"] = is_summary
            # We purposefully omit 'content' from this payload to save space in Vector DB,
            # unless the embedded text IS the content. 
            # If is_summary=False, text_to_embed IS content. 
            # Some vector stores don't store the embedding source text automatically?
            # LangChain `add_texts` keeps `texts`? 
            # Actually LangChain VectorStores usually store the text in `page_content`.
            # So we don't need to duplicate it in metadata.
            
            metadatas.append(payload)
            ids.append(seg.segment_id)

        self.vector_store.add_texts(
            texts=texts_to_embed,
            metadatas=metadatas,
            ids=ids
        )

    async def abuild(self, segments: List[Segment], batch_size: int = 100) -> None:
        """
        Asynchronous build execution.
        """
        if not segments:
            return

        logger.info(f"Async building index for {len(segments)} segments...")

        # 1. Async Store Segments
        pairs = [(s.segment_id, s) for s in segments]
        try:
            await self.doc_store.amset(pairs)
        except Exception as e:
            # Fallback for stores that might not implement async? 
            # BaseStore implements amset (it delegates to run_in_executor default or overridden)
            logger.error(f"Failed to async persist segments: {e}")
            raise e

        # 2. Async Vector Indexing
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            await self._process_vector_batch_async(batch)

    async def _process_vector_batch_async(self, batch: List[Segment]) -> None:
        texts_to_embed: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for seg in batch:
            summary = seg.metadata.get("summary")
            if summary:
                text = summary
                is_summary = True
            else:
                text = seg.content
                is_summary = False
            
            texts_to_embed.append(text)
            
            payload = seg.metadata.copy()
            payload["segment_id"] = seg.segment_id
            payload["type"] = seg.type.value
            payload["is_summary_embedding"] = is_summary
            
            metadatas.append(payload)
            ids.append(seg.segment_id)

        await self.vector_store.aadd_texts(
            texts=texts_to_embed,
            metadatas=metadatas,
            ids=ids
        )
