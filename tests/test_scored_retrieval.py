
import unittest
import uuid
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType

class TestScoredRetriever(unittest.TestCase):
    def test_scored_retrieval(self):
        print("\n--- Testing ScoredMultiVectorRetriever ---")
        
        # 1. Setup Stores
        doc_store = InMemoryStore()
        embeddings = FakeEmbeddings(size=10)
        # Use transient collection
        vector_store = Chroma(embedding_function=embeddings, collection_name=f"test_scored_{uuid.uuid4()}")
        
        # 2. Setup Data
        parent_id = str(uuid.uuid4())
        parent_doc = Document(page_content="Parent Content", metadata={"id": parent_id})
        doc_store.mset([(parent_id, parent_doc)])
        
        # 3 Chunks with varying similarity to query (simulated by content/embedding if possible, 
        # but Chroma FakeEmbeddings are random. We rely on search finding them.)
        # Ideally we'd mock the search, but let's try integration first.
        # Functionality: aggregate whatever scores are returned.
        
        chunk_1 = Document(page_content="High Match", metadata={"parent_id": parent_id})
        chunk_2 = Document(page_content="Low Match", metadata={"parent_id": parent_id})
        
        vector_store.add_documents([chunk_1, chunk_2])
        
        # 3. Create Retriever
        retriever = create_scored_dual_storage_retriever(
            vector_store=vector_store,
            doc_store=doc_store,
            id_key="parent_id",
            search_kwargs={"k": 10}, # Get both chunks
            search_type=SearchType.similarity_score_threshold,
        )
        
        # 4. Invoke
        # Since FakeEmbeddings are random, scores will be random but non-zero.
        results = retriever.invoke("High Match")
        
        self.assertTrue(results, "Should return scored documents")

        # 5. Assertions
        # Default mode restores dual-store hydration (dedup by parent).
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, "Parent Content")
        self.assertEqual(results[0].metadata.get("parent_id"), parent_id)

        for doc in results:
            self.assertIn("max_similarity_score", doc.metadata, "Each result should have max_similarity_score")
            self.assertIsInstance(doc.metadata["max_similarity_score"], float, "Score should be a float")
            self.assertIsNotNone(doc.id, "Retrieved document should have non-null id")

        print(f"Retrieved docs: {[doc.page_content for doc in results]}")
        print("--- Test Passed ---")

if __name__ == "__main__":
    unittest.main()
