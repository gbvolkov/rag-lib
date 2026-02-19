
import os
import argparse
import sys
import asyncio
from typing import List

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from rag_lib.config import Settings
from rag_lib.core.domain import Segment
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.loaders.csv_excel import CSVLoader
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.vectors.factory import create_vector_store
from rag_lib.core.store import LocalPickleStore
from rag_lib.core.index_builder import IndexBuilder
from langchain_community.embeddings import FakeEmbeddings # For quick testing
from langchain_openai import OpenAIEmbeddings

# Simple mock embedding for demo if no key
def get_embeddings():
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    print("WARNING: Using FakeEmbeddings (Random) for demo build.")
    return FakeEmbeddings(size=1536)

def main():
    parser = argparse.ArgumentParser(description="Build Dual Index: DocumentStore + VectorStore")
    parser.add_argument("--input", "-i", required=True, help="Input file (PDF/CSV)")
    parser.add_argument("--store-dir", "-s", default="storage/doc_store", help="Path for DocumentStore")
    parser.add_argument("--vector-provider", "-v", default="chroma", help="Vector Store Provider (chroma/qdrant)")
    parser.add_argument("--collection", "-c", default="rag_demo", help="Vector Collection Name")
    
    args = parser.parse_args()
    
    # 1. Load Documents
    print(f"Loading from {args.input}...")
    segments: List[Segment] = []
    
    if args.input.endswith(".pdf"):
        # We use stream backend for speed in demo, or lattice if needed
        loader = PDFLoader(args.input, backend="stream")
        segments = loader.load()
    elif args.input.endswith(".csv"):
        loader = CSVLoader(args.input)
        segments = loader.load()
    else:
        print(f"Unsupported file type: {args.input}")
        return

    print(f"Loaded {len(segments)} segments.")
    
    # 2. Chunking (Optional - simple logic here)
    # If segments are huge, we might re-chunk. 
    # For now assume Loader gives decent segments.
    # Note: SemanticChunker logic usually comes AFTER text extraction but BEFORE indexing.
    # If we wanted to semantic chunk, we would take raw text and chunk.
    # But Loaders return Segments.
    # Let's assume segments are ready for indexing.
    
    # 3. Setup Stores
    # Document Store
    os.makedirs(args.store_dir, exist_ok=True)
    doc_store_path = os.path.join(args.store_dir, "segments.pkl")
    doc_store = LocalPickleStore(doc_store_path)
    print(f"Initialized DocumentStore at {doc_store_path}")

    # Vector Store
    embeddings = get_embeddings()
    vector_store = create_vector_store(
        provider=args.vector_provider,
        embeddings=embeddings,
        collection_name=args.collection
    )
    print(f"Initialized VectorStore ({args.vector_provider})")
    
    # 4. Build Index
    builder = IndexBuilder(vector_store, doc_store)
    builder.build(segments)
    
    print("Done! Dual Index Created.")
    print(f" - Segments in {doc_store_path}")
    print(f" - Vectors in provider {args.vector_provider}")

if __name__ == "__main__":
    main()
