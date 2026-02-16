
import sys
import uuid
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from rag_lib.retrieval.composition import create_dual_storage_retriever

def test():
    print("Testing MultiVectorRetriever Dual Storage...")
    
    # 1. DocStore (Parents)
    doc_store = InMemoryStore()
    parent_id = str(uuid.uuid4())
    parent_doc = Document(page_content="Parent Content", metadata={"id": parent_id})
    doc_store.mset([(parent_id, parent_doc)])
    print(f"Stored parent {parent_id} in DocStore. Verify: {bool(doc_store.mget([parent_id])[0])}")
    
    # 2. VectorStore (Chunks)
    embeddings = FakeEmbeddings(size=10)
    vector_store = Chroma(embedding_function=embeddings, collection_name="test_collection")
    
    chunk_id = str(uuid.uuid4())
    chunk_doc = Document(
        page_content="Child Chunk", 
        metadata={"parent_id": parent_id} # Crucial link
    )
    vector_store.add_documents([chunk_doc], ids=[chunk_id])
    print(f"Stored chunk {chunk_id} in VectorStore with parent_id={parent_id}")
    
    # 3. Retriever
    retriever = create_dual_storage_retriever(
        vector_store=vector_store,
        docstore=doc_store,
        id_key="parent_id",
        search_kwargs={"k": 1}
    )
    
    # 4. Invoke
    results = retriever.invoke("Child Chunk")
    print(f"Results: {len(results)}")
    if results:
        print(f"Result 0 Content: {results[0].page_content}")
        print(f"Result 0 Metadata: {results[0].metadata}")
    else:
        print("FAILURE: No results retrieved.")

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"CRASH: {e}")
