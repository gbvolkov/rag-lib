from typing import List, Optional, Union, Any, Dict
from langchain_core.retrievers import BaseRetriever

# Robust Import Logic for LangChain variations
try:
    # Try langchain_classic first (as seen in user env)
    # Use top-level import as submodule import proved flaky in verification
    from langchain_classic.retrievers import EnsembleRetriever, MultiVectorRetriever
    from langchain_classic.retrievers import ContextualCompressionRetriever
    # Try to get CrossEncoderReranker from classic or standard
    try:
         from langchain.retrievers.document_compressors import CrossEncoderReranker
    except ImportError:
         try:
            from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
         except ImportError:
            from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
         
except ImportError:
    try:
        # Try standard langchain
        from langchain.retrievers import EnsembleRetriever, MultiVectorRetriever
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
    except ImportError:
         # Try specific submodules as last resort
        from langchain.retrievers.ensemble import EnsembleRetriever
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

# Try importing HuggingFaceCrossEncoder (optional dependency)
try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
except ImportError:
    HuggingFaceCrossEncoder = None

def create_ensemble_retriever(
    retrievers: List[BaseRetriever], 
    weights: Optional[List[float]] = None
) -> EnsembleRetriever:
    """
    Combines multiple retrievers into a single EnsembleRetriever using Reciprocal Rank Fusion (RRF).
    """
    if not retrievers:
        raise ValueError("Must provide at least one retriever for Ensemble.")
    
    if len(retrievers) == 1:
        # Optimization: return the single retriever if only one provided
        # BUT EnsembleRetriever returns a wrapper, which might be expected.
        # Let's return Ensemble to be consistent with type hint, or just the base.
        # Actually LangChain's EnsembleRetriever requires a list.
        pass

    return EnsembleRetriever(retrievers=retrievers, weights=weights)

def create_dual_storage_retriever(
    vector_store: VectorStore, 
    doc_store: BaseStore[str, Document], # Our Segment store usually stores Segments, but MultiVector expects compatible type
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
        byte_store=doc_store, # Parameter name is byte_store but accepts BaseStore
        id_key=id_key,
        search_kwargs=search_kwargs or {}
    )
    return retriever

def create_reranking_retriever(
    base_retriever_or_list: Union[BaseRetriever, List[BaseRetriever]],
    reranker_model: str = "BAAI/bge-reranker-base",
    top_n: int = 5,
    device: str = "cpu"
) -> ContextualCompressionRetriever:
    """
    Wraps a base retriever (or list of them) with a Cross-Encoder Reranker.
    
    Args:
        base_retriever_or_list: Single retriever or list (will be Auto-Ensembled).
        reranker_model: HuggingFace model name.
        top_n: Number of docs to return after reranking.
    """
    # 1. Handle List -> Ensemble
    if isinstance(base_retriever_or_list, list):
        base_retriever = create_ensemble_retriever(base_retriever_or_list)
    else:
        base_retriever = base_retriever_or_list
        
    # 2. Setup Reranker
    if HuggingFaceCrossEncoder is None:
        raise ImportError("langchain_community or sentence_transformers not installed.")
        
    model = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs={"device": device})
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    # 3. Create Compression Retriever
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
