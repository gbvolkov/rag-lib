import unittest
from unittest.mock import MagicMock
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from rag_lib.retrieval.scored_retriever import (
    ScoredMultiVectorRetriever,
    SearchType,
    HydrationMode,
)

class MockVectorStore(VectorStore):
    def __init__(self):
        self.mock_search = MagicMock()
        
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        pass
        
    def add_texts(self, texts, metadatas=None, **kwargs):
        pass
        
    def similarity_search(self, query, k=4, **kwargs):
        return []
        
    @property
    def embeddings(self):
        return None
        
    def similarity_search_with_relevance_scores(self, query, **kwargs):
        return self.mock_search(query, **kwargs)

class TestScoredRetrieverThreshold(unittest.TestCase):
    def test_threshold_passing(self):
        print("\n--- Testing Score Threshold Passing ---")
        
        # 1. Mock VectorStore
        mock_vectorstore = MockVectorStore()
        # Setup specific return for similarity_search_with_relevance_scores
        doc_high = Document(page_content="High", metadata={"pid": "p1"})
        doc_low = Document(page_content="Low", metadata={"pid": "p2"})
        
        mock_vectorstore.mock_search.return_value = [
            (doc_high, 0.9),
            (doc_low, 0.1)
        ]
        
        # 2. Setup DocStore
        doc_store = InMemoryStore()
        doc_store.mset([
            ("p1", Document(page_content="Parent 1")),
            ("p2", Document(page_content="Parent 2"))
        ])
        
        # 3. Create Retriever with Threshold
        threshold = 0.8
        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
            score_threshold=threshold
        )
        
        # 4. Invoke
        retriever.invoke("query")
        
        # 5. Verify Call Args
        # Verify that 'score_threshold' was passed in kwargs via mock_search
        args, kwargs = mock_vectorstore.mock_search.call_args
        print(f"Call kwargs: {kwargs}")
        
        self.assertIn("score_threshold", kwargs, "score_threshold should be in kwargs")
        self.assertEqual(kwargs["score_threshold"], threshold, "score_threshold should match")
        
        print("--- Threshold Passed correctly to VectorStore ---")

    def test_search_threshold_is_rejected(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()
        with self.assertRaises(ValueError):
            ScoredMultiVectorRetriever(
                vectorstore=mock_vectorstore,
                docstore=doc_store,
                id_key="pid",
                search_type=SearchType.similarity_score_threshold,
                search_threshold=0.5,
            )

    def test_missing_parent_keeps_chunk(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()  # intentionally empty

        orphan_chunk = Document(page_content="Orphan Chunk", metadata={"pid": "missing_parent"})
        mock_vectorstore.mock_search.return_value = [(orphan_chunk, 0.77)]

        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
            score_threshold=0.5,
        )

        results = retriever.invoke("query")

        self.assertEqual(len(results), 1, "Should keep the matched chunk")
        chunk = results[0]
        self.assertEqual(chunk.page_content, "Orphan Chunk")
        self.assertEqual(chunk.metadata["pid"], "missing_parent")
        self.assertIn("similarity_score", chunk.metadata)
        self.assertIn("max_similarity_score", chunk.metadata)
        self.assertEqual(chunk.id, "missing_parent")

    def test_returns_docs_with_and_without_parent_id(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()

        with_parent = Document(page_content="With Parent", metadata={"pid": "p1"})
        without_parent = Document(page_content="Without Parent", metadata={})
        mock_vectorstore.mock_search.return_value = [
            (with_parent, 0.8),
            (without_parent, 0.6),
        ]

        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
        )

        results = retriever.invoke("query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "With Parent")
        self.assertEqual(results[1].page_content, "Without Parent")
        self.assertEqual(results[0].metadata["pid"], "p1")
        self.assertNotIn("pid", results[1].metadata)
        self.assertIn("max_similarity_score", results[0].metadata)
        self.assertIn("max_similarity_score", results[1].metadata)

    def test_parents_replace_dedupes_and_uses_retrieved_children_scores_only(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()
        doc_store.mset([
            ("p1", Document(page_content="Parent 1", metadata={"id": "p1"})),
            ("p2", Document(page_content="Parent 2", metadata={"id": "p2"})),
        ])

        child_a = Document(page_content="A", metadata={"pid": "p1"})
        child_b = Document(page_content="B", metadata={"pid": "p1"})
        child_c = Document(page_content="C", metadata={"pid": "p2"})
        # Scores here represent the *retrieved children* only.
        mock_vectorstore.mock_search.return_value = [
            (child_a, 0.4),
            (child_b, 0.9),
            (child_c, 0.5),
        ]

        original_mget = doc_store.mget
        observed = {}

        def tracked_mget(ids):
            observed["ids"] = list(ids)
            return original_mget(ids)

        doc_store.mget = tracked_mget  # type: ignore[method-assign]

        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
            hydration_mode=HydrationMode.parents_replace,
        )

        results = retriever.invoke("query")
        self.assertEqual([d.page_content for d in results], ["Parent 1", "Parent 2"])
        self.assertEqual(observed.get("ids"), ["p1", "p2"])
        self.assertEqual(results[0].metadata["max_similarity_score"], 0.9)
        self.assertEqual(results[1].metadata["max_similarity_score"], 0.5)

    def test_children_plus_parents_order_and_parent_scores(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()
        doc_store.mset([
            ("p1", Document(page_content="Parent 1", metadata={"id": "p1"})),
            ("p2", Document(page_content="Parent 2", metadata={"id": "p2"})),
        ])

        child_a = Document(page_content="A", metadata={"pid": "p1"})
        child_b = Document(page_content="B", metadata={"pid": "p1"})
        child_c = Document(page_content="C", metadata={"pid": "p2"})
        mock_vectorstore.mock_search.return_value = [
            (child_a, 0.3),
            (child_b, 0.7),
            (child_c, 0.6),
        ]

        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
            hydration_mode=HydrationMode.children_plus_parents,
        )

        results = retriever.invoke("query")
        self.assertEqual(
            [d.page_content for d in results],
            ["A", "B", "C", "Parent 1", "Parent 2"],
        )
        self.assertEqual(results[3].metadata["max_similarity_score"], 0.7)
        self.assertEqual(results[4].metadata["max_similarity_score"], 0.6)

    def test_children_enriched_prefixes_parent_content(self):
        mock_vectorstore = MockVectorStore()
        doc_store = InMemoryStore()
        doc_store.mset([
            ("p1", Document(page_content="Parent 1", metadata={"id": "p1"})),
        ])

        child_a = Document(page_content="A", metadata={"pid": "p1"})
        mock_vectorstore.mock_search.return_value = [(child_a, 0.8)]

        retriever = ScoredMultiVectorRetriever(
            vectorstore=mock_vectorstore,
            docstore=doc_store,
            id_key="pid",
            search_type=SearchType.similarity_score_threshold,
            hydration_mode=HydrationMode.children_enriched,
        )

        results = retriever.invoke("query")
        self.assertEqual(len(results), 1)
        enriched = results[0]
        self.assertTrue(enriched.page_content.startswith("Parent 1"))
        self.assertIn("--- MATCHED CHILD CHUNK ---", enriched.page_content)
        self.assertEqual(enriched.metadata["parent_hydrated"], True)
        self.assertEqual(enriched.metadata["child_page_content_original"], "A")
        self.assertEqual(enriched.metadata["parent_page_content"], "Parent 1")

if __name__ == "__main__":
    unittest.main()
