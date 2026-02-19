import re
from typing import List, Optional, Any, Dict, Union
from rag_lib.core.domain import Segment
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.stores import BaseStore
from langchain_community.retrievers import BM25Retriever
from rapidfuzz import process, fuzz

class RegexRetriever(BaseRetriever):
    """
    Retriever that scans documents for regex patterns.
    Useful for ID lookups, code referencing, or specific format matching.
    """
    documents: List[Union[Document, Segment]]
    
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
            content = doc.content if isinstance(doc, Segment) else doc.page_content
            if pattern.search(content) or pattern.search(str(doc.metadata)):
                if isinstance(doc, Segment):
                    results.append(doc.to_langchain())
                else:
                    results.append(doc)
        
        return results

class FuzzyRetriever(BaseRetriever):
    """
    Retriever that uses approximate string matching (Levenshtein distance).
    Useful for typo-tolerant lookups of names, IDs, or short phrases.
    """
    documents: List[Union[Document, Segment]]
    threshold: int = 80 # 0-100 score
    mode: str = "partial_ratio" # "partial_ratio", "ratio", "token_set_ratio", "wratio"
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if not process:
            raise ImportError("rapidfuzz is required for FuzzyRetriever. Please install it.")
            
        scorers = {
            "partial_ratio": fuzz.partial_ratio,
            "ratio": fuzz.ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "wratio": fuzz.WRatio
        }
        
        scorer = scorers.get(self.mode.lower(), fuzz.partial_ratio)

        # We search against page_content or content
        choices = [doc.content if isinstance(doc, Segment) else doc.page_content for doc in self.documents]
        
        # Extract top matches
        # process.extract returns list of (match, score, index)
        matches = process.extract(
            query, 
            choices, 
            scorer=scorer, 
            limit=len(self.documents),
            score_cutoff=self.threshold
        )
        
        results = []
        for match, score, index in matches:
            doc = self.documents[index]
            if isinstance(doc, Segment):
                retrieved_doc = doc.to_langchain()
            else:
                # Create a copy to avoid mutating the original document in the store/list
                retrieved_doc = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
            
            # Inject score
            retrieved_doc.metadata["fuzzy_score"] = score
            results.append(retrieved_doc)
            
        return results

def create_vector_retriever(
    vector_store: VectorStore, 
    top_k: int = 4,
    search_type: str = "similarity", # "similarity", "mmr", "similarity_score_threshold"
    score_threshold: Optional[float] = None,
) -> BaseRetriever:
    """
    Factory validation for standard Vector Retriever.
    """
    kwargs = {"k": top_k}
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
        
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=kwargs
    )

def create_bm25_retriever(
    documents: List[Union[Document, Segment]],
    top_k: int = 4,
) -> BM25Retriever:
    """
    Factory for BM25 Retriever (In-Memory).
    Note: Requires all documents to be loaded in memory to build the index.
    """
    # Convert Segments to LangChain Documents for BM25Retriever compatibility
    lc_docs = []
    for doc in documents:
        if isinstance(doc, Segment):
            lc_docs.append(doc.to_langchain())
        else:
            lc_docs.append(doc)
            
    return BM25Retriever.from_documents(lc_docs, k=top_k)

def create_graph_retriever(
    vector_store: Optional[VectorStore],
    graph_store: Any,  # BaseGraphStore
    config: Optional[Any] = None,  # GraphQueryConfig
    embedder: Optional[Embeddings] = None,
    llm: Optional[BaseChatModel] = None,
    doc_store: Optional[BaseStore[str, Document]] = None,
    id_key: str = "segment_id",
) -> BaseRetriever:
    """
    Factory for Graph Retriever.
    """
    # Import locally to avoid circular imports or heavy deps if not used
    from rag_lib.retrieval.graph_retriever import GraphRetriever, GraphQueryConfig
    
    return GraphRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        config=config or GraphQueryConfig(),
        embedder=embedder,
        llm=llm,
        doc_store=doc_store,
        id_key=id_key,
    )

