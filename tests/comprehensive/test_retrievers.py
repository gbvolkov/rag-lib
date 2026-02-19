from unittest.mock import patch

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore

from rag_lib.retrieval.composition import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiVectorRetriever,
    create_dual_storage_retriever,
    create_ensemble_retriever,
    create_reranking_retriever,
)
from rag_lib.retrieval.retrievers import FuzzyRetriever, RegexRetriever

# Robust import BaseCrossEncoder
try:
    from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
except ImportError:
    try:
        from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder
    except ImportError:
        class BaseCrossEncoder:
            pass


class MockCrossEncoderModel(BaseCrossEncoder):
    def score(self, text_pairs):
        return [0.99] * len(text_pairs)


class StaticRetriever(BaseRetriever):
    docs: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.docs


class FixedVectorStore(VectorStore):
    def __init__(self):
        self._docs: list[Document] = []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        store = cls()
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store

    def add_texts(self, texts, metadatas=None, **kwargs):
        ids = kwargs.get("ids", [])
        metadatas = metadatas or [{} for _ in texts]
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            if idx < len(ids):
                metadata = metadata.copy()
                metadata.setdefault("id", ids[idx])
            self._docs.append(Document(page_content=text, metadata=metadata))
        return ids

    def similarity_search(self, query, k=4, **kwargs):
        return self._docs[:k]

    def similarity_search_with_relevance_scores(self, query, k=4, **kwargs):
        return [(doc, 1.0) for doc in self._docs[:k]]

    @property
    def embeddings(self):
        return None


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="The specific ID is ABC-123",
            metadata={"id": "doc1", "full_content": "Full Content 1"},
        ),
        Document(
            page_content="Concept of RAG retrieval",
            metadata={"id": "doc2", "full_content": "Full Content 2"},
        ),
        Document(
            page_content="Typos: Concpet of RGA retreival",
            metadata={"id": "doc3", "full_content": "Full Content 3"},
        ),
    ]


def test_kitchen_sink_components(sample_docs):
    vector = StaticRetriever(docs=[sample_docs[1]])
    regex = RegexRetriever(documents=sample_docs)
    try:
        import rapidfuzz  # noqa: F401

        fuzzy: BaseRetriever = FuzzyRetriever(documents=sample_docs, threshold=50)
    except ImportError:
        fuzzy = StaticRetriever(docs=[sample_docs[2]])

    ensemble = create_ensemble_retriever([vector, regex, fuzzy])
    assert isinstance(ensemble, EnsembleRetriever)

    model = MockCrossEncoderModel()
    with patch("rag_lib.retrieval.composition.HuggingFaceCrossEncoder", return_value=model):
        reranked = create_reranking_retriever(ensemble)
    assert isinstance(reranked, ContextualCompressionRetriever)


def test_dual_storage_hydration_with_custom_id_key():
    vector_store = FixedVectorStore()
    vector_store.add_texts(
        ["Summary about retrieval"],
        metadatas=[{"id": "doc2"}],
        ids=["chunk-1"],
    )

    doc_store = InMemoryStore()
    doc_store.mset(
        [
            (
                "doc2",
                Document(
                    page_content="FULL: Concept of RAG retrieval",
                    metadata={"id": "doc2"},
                ),
            )
        ]
    )

    dual = create_dual_storage_retriever(
        vector_store=vector_store,
        doc_store=doc_store,
        id_key="id",
        search_kwargs={"k": 1},
    )
    assert isinstance(dual, MultiVectorRetriever)

    results = dual.invoke("retrieval")
    assert len(results) == 1
    assert results[0].metadata["id"] == "doc2"
    assert results[0].page_content.startswith("FULL:")
