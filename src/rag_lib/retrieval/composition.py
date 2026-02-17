from typing import List, Optional, Any, Dict, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore

# Use strict imports from langchain_classic as per environment
try:
    from langchain_classic.retrievers import (
        EnsembleRetriever, 
        MultiVectorRetriever,
        ContextualCompressionRetriever
    )
    from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
except ImportError as e:
    # Fallback/Error handling if classic is missing (unlikely given env) but user wanted NO fallback
    raise ImportError(f"Required 'langchain-classic' not found or incomplete: {e}")

# Note: SearchType is not available in classic, so we use string literals/defaults.
from .cross_encoder_reranker_with_score import TournamentCrossEncoderReranker

# Try importing HuggingFaceCrossEncoder (optional dependency)
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:
    HuggingFaceCrossEncoder = None

def create_ensemble_retriever(
    retrievers: List[BaseRetriever],
    weights: Optional[List[float]] = None
) -> BaseRetriever:
    """
    Creates an EnsembleRetriever from a list of retrievers.
    """
    if not retrievers:
        raise ValueError("Must provide at least one retriever")
        
    return EnsembleRetriever(
        retrievers=retrievers,
        weights=weights
    )

def create_dual_storage_retriever(
    vector_store: VectorStore, 
    docstore: BaseStore[str, Document], # Renamed to standard docstore
    id_key: str = "segment_id",
    search_kwargs: Optional[Dict[str, Any]] = None
) -> MultiVectorRetriever:
    """
    Creates a MultiVectorRetriever (Dual Storage) which uses:
    1. Vector Store for search (using lightweight keys/embeddings).
    2. Document Store for content lookup (using 'id_key').
    
    Note: 'doc_store' must implement mget/mset.
    If 'doc_store' stores 'Segment' objects, MultiVectorRetriever will return 'Segment' objects (as Docs?).
    LangChain expects mget to return things that can be treated as Documents or bytes?
    Actually MultiVectorRetriever just returns whatever doc_store.mget returns.
    """

    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=docstore, 
        id_key=id_key,
        search_kwargs=search_kwargs or {},
        search_type="similarity_score_threshold"
    )
    return retriever

from rag_lib.retrieval.scored_retriever import ScoredMultiVectorRetriever, SearchType

def create_scored_dual_storage_retriever(
    vector_store: VectorStore,
    docstore: BaseStore[str, Document],
    id_key: str = "segment_id",
    search_kwargs: Optional[Dict[str, Any]] = None,
    search_type: SearchType = SearchType.similarity,
    score_threshold: float | None = None
) -> BaseRetriever:
    """
    Creates a ScoredMultiVectorRetriever.
    Like Dual Storage, but returns Parent Documents with aggregated Similarity Scores (max) from chunks.
    Metadata 'max_similarity_score' is added to the parent document.
    """
    return ScoredMultiVectorRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        id_key=id_key,
        search_kwargs=search_kwargs or {},
        search_type=search_type,
        score_threshold=score_threshold
    )

def create_reranking_retriever(
    base_retriever_or_list: Union[BaseRetriever, List[BaseRetriever]],
    reranker_model: Union[str, BaseCrossEncoder] = "BAAI/bge-reranker-base",
    top_n: int = 5,
    max_score_ratio: float = 0.0, #ratio of maximum score value
    device: str = "cpu"
) -> ContextualCompressionRetriever:
    """
    Wraps a base retriever (or list of them) with a Cross-Encoder Reranker.
    
    Args:
        base_retriever_or_list: Single retriever or list (will be Auto-Ensembled).
        reranker_model: HuggingFace model name or a pre-built BaseCrossEncoder.
        top_n: Number of docs to return after reranking.
    """
    # 1. Handle List -> Ensemble
    if isinstance(base_retriever_or_list, list):
        base_retriever = create_ensemble_retriever(base_retriever_or_list)
    else:
        base_retriever = base_retriever_or_list

    # 2. Setup Reranker
    if isinstance(reranker_model, str):
        if HuggingFaceCrossEncoder is None:
            raise ImportError("langchain_community or sentence_transformers not installed.")
        model = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={"trust_remote_code": True, "device": device},
        )
    else:
        model = reranker_model

    compressor = TournamentCrossEncoderReranker(
        model=model, top_n=top_n, tournament_size=10, min_ratio=max_score_ratio
    )
    # 3. Create Compression Retriever
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

def create_graph_hybrid_retriever(
    vector_retriever: BaseRetriever,
    graph_retriever: BaseRetriever,
    weights: List[float] = [0.7, 0.3]
) -> EnsembleRetriever:
    """
    Combines Vector Retrieval (Semantic) with Graph Retrieval (Structural/Relational).
    Default weight favors Vector (0.7) over Graph (0.3).
    """
    return create_ensemble_retriever(
        retrievers=[vector_retriever, graph_retriever],
        weights=weights
    )

