import unittest
from unittest.mock import MagicMock
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from rag_lib.retrieval.scored_retriever import ScoredMultiVectorRetriever, SearchType

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
        print("\n--- Testing Search Threshold Passing ---")
        
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
            search_threshold=threshold
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

if __name__ == "__main__":
    unittest.main()
