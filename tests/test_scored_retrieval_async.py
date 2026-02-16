
import unittest
import asyncio
import uuid
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType

class TestScoredRetrieverAsync(unittest.IsolatedAsyncioTestCase):
    async def test_scored_retrieval_async(self):
        print("\n--- Testing ScoredMultiVectorRetriever ASYNC ---")
        
        # 1. Setup Stores
        doc_store = InMemoryStore()
        embeddings = FakeEmbeddings(size=10)
        vector_store = Chroma(embedding_function=embeddings, collection_name=f"test_scored_async_{uuid.uuid4()}")
        
        # 2. Setup Data
        parent_id = str(uuid.uuid4())
        parent_doc = Document(page_content="Async Parent Content", metadata={"id": parent_id})
        await doc_store.amset([(parent_id, parent_doc)])
        
        chunk_1 = Document(page_content="Async High Match", metadata={"parent_id": parent_id})
        await vector_store.aadd_documents([chunk_1])
        
        # 3. Create Retriever
        retriever = create_scored_dual_storage_retriever(
            vector_store=vector_store,
            docstore=doc_store,
            id_key="parent_id",
            search_kwargs={"k": 2},
            search_type=SearchType.similarity_score_threshold,
            search_threshold=0.0 # simple threshold
        )
        
        # 4. Invoke Async
        results = await retriever.ainvoke("Async High Match")
        
        self.assertTrue(results, "Should return at least one parent")
        retrieved_doc = results[0]
        
        print(f"Retrieved Parent Metadata: {retrieved_doc.metadata}")
        
        # 5. Assertions
        self.assertIn("max_similarity_score", retrieved_doc.metadata, "Metadata should contain 'max_similarity_score'")
        self.assertIsInstance(retrieved_doc.metadata["max_similarity_score"], float, "Score should be a float")
        self.assertEqual(retrieved_doc.page_content, "Async Parent Content", "Should retrieve correct parent")
        
        print(f"Aggregated Score: {retrieved_doc.metadata['max_similarity_score']}")
        print("--- Async Test Passed ---")

if __name__ == "__main__":
    unittest.main()
