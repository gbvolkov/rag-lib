from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.vectorstores import VectorStore

from rag_lib.retrieval.scored_retriever import ScoredMultiVectorRetriever


class MockVectorStore(VectorStore):
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


if __name__ == "__main__":
    print("ScoredMultiVectorRetriever fields:", ScoredMultiVectorRetriever.model_fields.keys())

    retriever = ScoredMultiVectorRetriever(
        vector_store=MockVectorStore(),
        doc_store=InMemoryStore(),
        id_key="doc_id",
    )
    print("Created retriever with canonical names:", isinstance(retriever, ScoredMultiVectorRetriever))