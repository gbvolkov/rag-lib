import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup Env EARLY (Before any library imports that might trigger config loading)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results
# 0. Setup Environment FIRST (before library imports that might read env)
setup_environment()

# 1. Imports - Library Abstractions ONLY (Client View)
from rag_lib.core.domain import Segment
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.vectors.factory import create_vector_store
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.llm.factory import create_llm

"""
E2E Example 02: Markdown Enrichment + Vector Search Workflow
...
"""

def main():
    print_section("02. Markdown Enrichment + Vector Search")
    # ... rest of main

    # 2. Load
    md_path = Path(__file__).parent.parent / "docs" / "quotes.toscrape.com_index.md"
    print(f"Loading {md_path}...")
    
    from rag_lib.loaders.data_loaders import TextLoader
    loader = TextLoader(str(md_path))
    docs = loader.load()

    # 3. Chunk
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    segments = chunker.split_documents(docs)
    segments = segments[:5] # Demo limit
    save_json_results(segments, "02_markdown_enrichment_vector", "raw_segments")
    
    # 4. Enrich
    print("Enriching segments...")
    llm = create_llm(model_name="gpt-4.1-nano", provider="openai")
    enricher = SegmentEnricher(llm)
    enriched = enricher.enrich(segments)
    save_json_results(enriched, "02_markdown_enrichment_vector", "enriched_segments")
    print(f"Enriched Metadata Sample: {enriched[0].metadata}")

    # 5. Index (Vector - FAISS)
    print("\nIndexing into FAISS Vector Store...")
    try:
        # Use Factories
        embeddings = create_embeddings_model(provider="openai")
        # Note: factory returns a fresh store (potentially with a dummy doc)
        vector_store = create_vector_store(provider="faiss", embeddings=embeddings)
        
        # Use Indexer
        # Indexer handles adding segments to the store
        indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
        indexer.index(enriched)
        
        # 6. Retrieve
        print("Retrieving using VectorRetriever...")
        retriever = create_vector_retriever(vector_store=vector_store, top_k=2)
        
        query = "einstein quotes"
        results = retriever.invoke(query)
        
        print(f"Query: {query}")
        for r in results:
            # Note: r is a Document (converted from Segment by Indexer/Retriever flow)
            print(f"Match: {r.page_content[:50]}...")

        save_json_results(results, "02_markdown_enrichment_vector", "retrieved_results")
        
    except Exception as e:
        print(f"Vector pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
