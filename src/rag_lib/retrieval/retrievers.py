import re
from typing import List, Optional, Any, Dict
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

class RegexRetriever(BaseRetriever):
    """
    Retriever that scans documents for regex patterns.
    Useful for ID lookups, code referencing, or specific format matching.
    """
    documents: List[Document]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Finds documents where the query is found as a regex pattern match.
        OR, if the query itself is a pattern, finds documents matching it.
        
        For simplicity in RAG flow: 
        1. We assume the 'query' is the pattern we are looking FOR in the documents.
        2. OR the 'query' is a specific key (like an ID) and we use a pre-configured pattern to find it.
        
        Refined Logic:
        - If the retriever was initialized with a specific 'pattern_template' (e.g. r"ID-{query}"), 
          it uses that template with the query inserted.
        - Otherwise, it treats the query itself as a literal string to find (re.escape) or raw regex.
        """
        results = []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE) 
        except re.error:
            # Fallback for invalid regex queries
            return []
            
        for doc in self.documents:
            if pattern.search(doc.page_content) or pattern.search(str(doc.metadata)):
                results.append(doc)
        
        return results

class FuzzyRetriever(BaseRetriever):
    """
    Retriever that uses approximate string matching (Levenshtein distance).
    Useful for typo-tolerant lookups of names, IDs, or short phrases.
    """
    documents: List[Document]
    threshold: int = 80 # 0-100 score
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not process:
            raise ImportError("rapidfuzz is required for FuzzyRetriever. Please install it.")
            
        # We search against page_content
        choices = [doc.page_content for doc in self.documents]
        
        # Extract top matches
        # process.extract returns list of (match, score, index)
        matches = process.extract(
            query, 
            choices, 
            scorer=fuzz.partial_ratio, 
            limit=len(self.documents),
            score_cutoff=self.threshold
        )
        
        results = []
        for match, score, index in matches:
            results.append(self.documents[index])
            
        return results

def get_vector_retriever(
    vector_store: VectorStore, 
    k: int = 4,
    search_type: str = "similarity", # "similarity", "mmr", "similarity_score_threshold"
    score_threshold: Optional[float] = None
) -> BaseRetriever:
    """
    Factory validation for standard Vector Retriever.
    """
    kwargs = {"k": k}
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
        
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=kwargs
    )

def get_bm25_retriever(
    documents: List[Document],
    k: int = 4
) -> BM25Retriever:
    """
    Factory for BM25 Retriever (In-Memory).
    Note: Requires all documents to be loaded in memory to build the index.
    """
    return BM25Retriever.from_documents(documents, k=k)

def get_graph_retriever(
    graph_store: Any, # BaseGraphStore
    search_depth: int = 1
) -> BaseRetriever:
    """
    Factory for Graph Retriever.
    """
    # Import locally to avoid circular imports or heavy deps if not used
    from rag_lib.retrieval.graph_retriever import GraphRetriever
    
    return GraphRetriever(store=graph_store, search_depth=search_depth)

