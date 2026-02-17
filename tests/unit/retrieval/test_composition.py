import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore
# Import classes from the module under test to ensure type matching
from rag_lib.retrieval.composition import (
    EnsembleRetriever, 
    MultiVectorRetriever, 
    ContextualCompressionRetriever,
    create_ensemble_retriever,
    create_dual_storage_retriever,
    create_reranking_retriever
)

# Mock Classes
class MockRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return []


class DummyVectorStore(VectorStore):
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        return cls()

    def add_texts(self, texts, metadatas=None, **kwargs):
        return []

    def similarity_search(self, query, k=4, **kwargs):
        return []

    @property
    def embeddings(self):
        return None

def test_ensemble_creation():
    r1 = MockRetriever()
    r2 = MockRetriever()
    
    ensemble = create_ensemble_retriever([r1, r2], weights=[0.5, 0.5])
    assert isinstance(ensemble, EnsembleRetriever)
    assert len(ensemble.retrievers) == 2
    assert ensemble.weights == [0.5, 0.5]

def test_dual_storage_creation():
    vector_store = DummyVectorStore()
    doc_store = InMemoryStore()
    
    dual = create_dual_storage_retriever(vector_store, doc_store, id_key="seg_id")
    assert isinstance(dual, MultiVectorRetriever)
    assert dual.vectorstore == vector_store
    assert dual.docstore == doc_store
    assert dual.id_key == "seg_id"

# Robustly import BaseCrossEncoder for mocking
try:
    from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
except ImportError:
    try:
        from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
    except ImportError:
        # Fallback if we can't find it (unlikely if CrossEncoderReranker works)
        class BaseCrossEncoder: pass

class MockCrossEncoderModel(BaseCrossEncoder):
    def score(self, text_pairs):
        # Return arbitrary scores matching length of pairs
        return [0.9] * len(text_pairs)

def test_reranking_creation_simple():
    r1 = MockRetriever()
    model = MockCrossEncoderModel()
    
    # We pass the model object directly if our factory supports it?
    # Wait, create_reranking_retriever(..., reranker_model="name")
    # It instantiates HuggingFaceCrossEncoder(model_name="name") inside.
    # We can't pass an OBJECT to create_reranking_retriever currently.
    # We MUST patch HuggingFaceCrossEncoder to return our MockCrossEncoderModel.
    
    from unittest.mock import patch
    with patch("rag_lib.retrieval.composition.HuggingFaceCrossEncoder", return_value=model):
         reranked = create_reranking_retriever(r1, top_n=3)
         assert isinstance(reranked, ContextualCompressionRetriever)
         assert reranked.base_retriever == r1

def test_reranking_auto_ensemble():
    model = MockCrossEncoderModel()
    from unittest.mock import patch
    with patch("rag_lib.retrieval.composition.HuggingFaceCrossEncoder", return_value=model):
        r1 = MockRetriever()
        r2 = MockRetriever()
        
        reranked = create_reranking_retriever([r1, r2])
        
        assert isinstance(reranked, ContextualCompressionRetriever)
        assert isinstance(reranked.base_retriever, EnsembleRetriever)
