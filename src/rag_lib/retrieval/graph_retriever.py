from typing import List, Optional, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from rag_lib.graph.store import BaseGraphStore
from rag_lib.core.logger import logger

class GraphRetriever(BaseRetriever):
    """
    Retriever that uses a GraphStore to find relevant segments.
    Supports Local (subgraph) and Global (community summary) retrieval modes.
    """
    store: BaseGraphStore
    vector_store: Optional[VectorStore] = None
    mode: str = "local"  # "local", "global"
    search_depth: int = 1

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        logger.info(f"Graph Retrieval query='{query}' mode='{self.mode}'")
        
        if self.mode == "global":
            return self._global_search(query)
        else:
            return self._local_search(query)

    def _global_search(self, query: str) -> List[Document]:
        """
        Global Search: Retrieves community summaries relevant to the query.
        Requires vector_store to be populated with community summaries.
        """
        if not self.vector_store:
            logger.warning("Global search requested but no vector_store provided.")
            return []
            
        # Search for segments marked as community summaries
        # We assume the Indexer has stored them with metadata `is_community_summary=True`
        # and likely `type="community_summary"`
        
        try:
            # Note: filter syntax depends on the specific VectorStore implementation (Chroma, Qdrant, etc.)
            # Here we try a generic filter that might work for Chroma/Qdrant if they support dict filters.
            # If not, we might need a specific filter object. 
            # ideally we'd use self.vector_store.as_retriever(search_kwargs={"filter": ...})
            # but we want direct control here.
            
            # For robustness, we'll try to use a filter if possible, or post-filter if not expensive.
            # But global search relies on vector similarity of the summary significantly.
            
            k = 10 # Number of communities to retrieve
            
            # Simple approach: search and filter. 
            # Better approach: pass filter to similarity_search.
            # We'll assume a dict filter works for now (Chroma/Qdrant compatible).
            results = self.vector_store.similarity_search(
                query, 
                k=k, 
                filter={"is_community_summary": True}
            )
            
            logger.info(f"Global search found {len(results)} community summaries.")
            return results
            
        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return []

    def _local_search(self, query: str) -> List[Document]:
        # 1. Start nodes: Keyword search on the graph
        start_nodes = self.store.search_nodes(query)
        logger.debug(f"Found {len(start_nodes)} starting nodes")

        relevant_segments_ids = set()
        
        # 2. Traverse neighbors
        results = []
        for node in start_nodes:
             content = f"Entity: {node.label}\nDescription: {node.description}\nType: {node.type}"
             results.append(Document(page_content=content, metadata={"node_id": node.id, "source": "graph"}))
             
             neighbors = self.store.get_neighbors(node.id, depth=self.search_depth)
             for n in neighbors:
                 content = f"Related Entity: {n.label}\nRelation: Linked to {node.label}\nDescription: {n.description}"
                 results.append(Document(page_content=content, metadata={"node_id": n.id, "source": "graph_neighbor"}))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        logger.info(f"Async Graph Retrieval query='{query}' mode='{self.mode}'")
        
        if self.mode == "global":
            return await self._aglobal_search(query)
        else:
            return await self._alocal_search(query)

    async def _aglobal_search(self, query: str) -> List[Document]:
        if not self.vector_store:
            logger.warning("Global search requested but no vector_store provided.")
            return []
            
        try:
            k = 10
            # Use async similarity search if available available
            results = await self.vector_store.asimilarity_search(
                query, 
                k=k, 
                filter={"is_community_summary": True}
            )
            logger.info(f"Async global search found {len(results)} community summaries.")
            return results
        except Exception as e:
            logger.error(f"Async global search failed: {e}")
            return []

    async def _alocal_search(self, query: str) -> List[Document]:
        # 1. Start nodes: Keyword search on the graph
        start_nodes = await self.store.asearch_nodes(query)
        logger.debug(f"Found {len(start_nodes)} starting nodes")

        results = []
        for node in start_nodes:
             content = f"Entity: {node.label}\nDescription: {node.description}\nType: {node.type}"
             results.append(Document(page_content=content, metadata={"node_id": node.id, "source": "graph"}))
             
             neighbors = await self.store.aget_neighbors(node.id, depth=self.search_depth)
             for n in neighbors:
                 content = f"Related Entity: {n.label}\nRelation: Linked to {node.label}\nDescription: {n.description}"
                 results.append(Document(page_content=content, metadata={"node_id": n.id, "source": "graph_neighbor"}))

        logger.info(f"Retrieved {len(results)} graph context documents")
        return results
