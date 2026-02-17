
import unittest
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
            search_type=SearchType.similarity_score_threshold
        )
        
        # 4. Invoke Async
        results = await retriever.ainvoke("Async High Match")
        
        self.assertTrue(results, "Should return scored documents")
        retrieved_doc = results[0]

        # 5. Assertions
        self.assertEqual(retrieved_doc.page_content, "Async Parent Content", "Should hydrate parent by default")
        self.assertEqual(retrieved_doc.metadata.get("parent_id"), parent_id)
        self.assertIn("max_similarity_score", retrieved_doc.metadata, "Metadata should contain 'max_similarity_score'")
        self.assertIsInstance(retrieved_doc.metadata["max_similarity_score"], float, "Score should be a float")
        self.assertIsNotNone(retrieved_doc.id, "Retrieved document should have non-null id")

        print(f"Aggregated Score: {retrieved_doc.metadata['max_similarity_score']}")
        print("--- Async Test Passed ---")

    async def test_scored_retrieval_async_missing_parent_keeps_chunk(self):
        doc_store = InMemoryStore()  # intentionally empty
        embeddings = FakeEmbeddings(size=10)
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name=f"test_scored_async_orphan_{uuid.uuid4()}",
        )

        orphan_chunk = Document(
            page_content="Async Orphan Chunk",
            metadata={"parent_id": "missing_parent"},
        )
        await vector_store.aadd_documents([orphan_chunk])

        retriever = create_scored_dual_storage_retriever(
            vector_store=vector_store,
            docstore=doc_store,
            id_key="parent_id",
            search_kwargs={"k": 2},
            search_type=SearchType.similarity_score_threshold,
        )

        results = await retriever.ainvoke("Async Orphan Chunk")
        self.assertTrue(results)
        fallback = results[0]
        self.assertEqual(fallback.page_content, "Async Orphan Chunk")
        self.assertEqual(fallback.metadata["parent_id"], "missing_parent")
        self.assertIn("similarity_score", fallback.metadata)
        self.assertIn("max_similarity_score", fallback.metadata)
        self.assertIsNotNone(fallback.id, "Chunk should have non-null id")

if __name__ == "__main__":
    unittest.main()
