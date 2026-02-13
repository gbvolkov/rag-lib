import argparse
import sys
import logging
from typing import List, Optional
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import rag_lib components
try:
    from rag_lib.retrieval.retrievers import (
        RegexRetriever, 
        FuzzyRetriever,
        get_bm25_retriever,
        get_vector_retriever
    )
    from rag_lib.retrieval.composition import (
        create_ensemble_retriever,
        create_reranking_retriever,
        create_dual_storage_retriever
    )
    # Assume we have a way to load stores (mocked for this script as we don't have real data persistence layer ready in this phase)
    # In real usage, these would come from rag_lib.core.store or similar.
except ImportError as e:
    logger.error(f"Failed to import rag_lib components: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Query the RAG Index using Advanced Retrievers.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument("--mode", type=str, choices=["vector", "bm25", "regex", "fuzzy", "ensemble", "rerank", "dual"], default="vector", help="Retrieval mode.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    
    args = parser.parse_args()
    
    query = args.query
    mode = args.mode
    top_k = args.top_k
    
    logger.info(f"Searching for: '{query}' using mode: {mode}")
    
    # MOCK DATA LAYER (Since we don't have a populated DB for this phase verify)
    # In a real scenario, we would load existing stores.
    docs = [
        Document(page_content="RAG systems combine retrieval and generation.", metadata={"id": "doc1"}),
        Document(page_content="Vector databases store embeddings for similarity search.", metadata={"id": "doc2"}),
        Document(page_content="BM25 is a ranking function used in information retrieval.", metadata={"id": "doc3"}),
        Document(page_content="LangChain provides modules for building LLM apps.", metadata={"id": "doc4"}),
        Document(page_content="Specific Error Code: ERR-999 caused system crash.", metadata={"id": "doc5"}),
        Document(page_content="Typo: LangChian is gret.", metadata={"id": "doc6"})
    ]
    
    # -- Initialize Retrievers --
    retriever = None
    
    if mode == "regex":
        retriever = RegexRetriever(documents=docs)
        
    elif mode == "fuzzy":
        try:
             import rapidfuzz
             retriever = FuzzyRetriever(documents=docs, threshold=60)
        except ImportError:
             logger.warning("rapidfuzz not installed. Using basic fuzzy mock (finding nothing).")
             retriever = RegexRetriever(documents=[]) # Mock fallthrough
             
    elif mode == "bm25":
        retriever = get_bm25_retriever(docs)
        
    elif mode == "vector":
        # Check for vector store dependencies
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=768)
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = get_vector_retriever(vectorstore)
        except ImportError:
             logger.error("FAISS or Embeddings not available. Cannot use vector mode.")
             sys.exit(1)
             
    elif mode == "ensemble":
        # Combine BM25 and Vector (Simulated)
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=768)
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            vec_retriever = get_vector_retriever(vectorstore)
            bm25_retriever = get_bm25_retriever(docs)
            
            retriever = create_ensemble_retriever([vec_retriever, bm25_retriever], weights=[0.5, 0.5])
        except ImportError:
             logger.error("Dependencies missing for Ensemble.")
             sys.exit(1)
             
    elif mode == "rerank":
        # BM25 + Reranker
        bm25_retriever = get_bm25_retriever(docs)
        # Note: This requires sentence-transformers/torch
        try:
            retriever = create_reranking_retriever(bm25_retriever, top_n=top_k)
        except Exception as e:
            logger.error(f"Failed to init Reranker (likely missing model/libs): {e}")
            sys.exit(1)

    elif mode == "dual":
        # Simulated Dual Storage
        # Vector + DocStore
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import FakeEmbeddings
            from langchain_core.stores import InMemoryStore
            
            embeddings = FakeEmbeddings(size=768)
            # Index summaries (mocked as docs content for now)
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            # Doc Store (Has "FULL" content)
            doc_store = InMemoryStore()
            full_docs = {
                d.metadata["id"]: Document(page_content=f"FULL CONTENT: {d.page_content}", metadata=d.metadata)
                for d in docs
            }
            doc_store.mset(list(full_docs.items()))
            
            retriever = create_dual_storage_retriever(vectorstore, doc_store, id_key="id")
            
        except ImportError:
             logger.error("Dependencies missing for Dual Storage.")
             sys.exit(1)
             
    # -- Execution --
    if retriever:
        try:
            results = retriever.invoke(query)
            logger.info(f"\n--- Results ({len(results)}) ---")
            for i, doc in enumerate(results[:top_k]):
                logger.info(f"[{i+1}] {doc.page_content} (Meta: {doc.metadata})")
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
