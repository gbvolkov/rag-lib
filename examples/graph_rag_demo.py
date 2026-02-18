import asyncio
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.chat_models import FakeListChatModel

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.retrieval.graph_retriever import GraphQueryConfig, GraphRetriever


class InMemoryLexicalVectorStore(VectorStore):
    def __init__(self, docs: list[Document] | None = None):
        self.docs = docs or []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = kwargs.get("ids", [None for _ in texts])
        docs = []
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata.setdefault("segment_id", ids[idx])
            docs.append(Document(page_content=text, metadata=metadata))
        return cls(docs=docs)

    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [None for _ in texts]
        for idx, text in enumerate(texts):
            metadata = dict(metadatas[idx] if idx < len(metadatas) else {})
            if idx < len(ids) and ids[idx] is not None:
                metadata.setdefault("segment_id", ids[idx])
            self.docs.append(Document(page_content=text, metadata=metadata))
        return ids

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        rows = self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in rows]

    def similarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        query_terms = set(query.lower().split())
        rows = []
        for doc in self.docs:
            metadata = doc.metadata or {}
            if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                continue
            text_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)
            rows.append((doc, score))
        rows.sort(key=lambda row: (-row[1], str((row[0].metadata or {}).get("segment_id", "")), row[0].page_content))
        return rows[:k]

    async def asimilarity_search_with_relevance_scores(self, query, k=4, filter=None, **kwargs):
        return self.similarity_search_with_relevance_scores(query, k=k, filter=filter, **kwargs)

    def get_by_ids(self, ids):
        wanted = {str(value) for value in ids}
        return [doc for doc in self.docs if str((doc.metadata or {}).get("segment_id", "")) in wanted]

    async def aget_by_ids(self, ids):
        return self.get_by_ids(ids)

    @property
    def embeddings(self):
        return None


def run_demo():
    print("--- Starting Graph RAG Verification ---")

    # 1. Setup Data
    segment = Segment(content="Albert Einstein developed Relativity.", segment_id="seg1", type=SegmentType.TEXT)

    # 2. Setup Components
    mock_response = json.dumps(
        {
            "entities": [
                {"name": "Albert Einstein", "type": "Person", "description": "Physicist"},
                {"name": "Relativity", "type": "Concept", "description": "Theory"},
            ],
            "relationships": [
                {"source": "Albert Einstein", "target": "Relativity", "relation": "DEVELOPED", "weight": 1.0}
            ],
        }
    )
    llm = FakeListChatModel(responses=[mock_response])

    store = NetworkXGraphStore()
    extractor = EntityExtractor(llm=llm, store=store)

    vector_store = InMemoryLexicalVectorStore(
        docs=[Document(page_content=segment.content, metadata={"segment_id": segment.segment_id})]
    )

    # Initialize Indexer with EntityExtractor
    indexer = Indexer(vector_store=vector_store, embeddings=None, entity_extractor=extractor)

    # 3. Run Integrated Indexing
    print("--- Running Indexer (Triggering Graph Extraction) ---")
    indexer.index([segment])

    # 4. Verify Graph Content
    print(f"Graph initialized with nodes: {', '.join([n.label for n in store.search_nodes('')])}")

    # 5. Verify Retrieval
    print("\n--- Testing Retrieval for 'Einstein' ---")
    retriever = GraphRetriever(
        vector_store=vector_store,
        graph_store=store,
        config=GraphQueryConfig(mode="local", max_hops=1, enable_keyword_extraction=False),
    )
    results = retriever.invoke("Einstein")

    output = [
        doc.metadata.get("entity_id") or doc.metadata.get("edge_id") or doc.metadata.get("source_segment_id")
        for doc in results
    ]

    print(f"Retrieved evidence IDs: {output}")

    if any(item == "Albert Einstein" for item in output) and any(item == "Relativity" for item in output):
        print("SUCCESS: Retrieved both the primary node and its neighbor.")
    else:
        print("WARNING: Did not retrieve both expected nodes.")


if __name__ == "__main__":
    run_demo()
