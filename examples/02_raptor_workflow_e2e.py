import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Dependency checks for RAPTOR
from example_utils import _check_install, setup_environment, print_section, load_pdf_text

# Ensure RAPTOR deps
try:
    import umap
except ImportError:
    _check_install("umap-learn", "umap")
try:
    import sklearn
except ImportError:
    _check_install("scikit-learn", "sklearn")
try:
    import numpy
except ImportError:
    _check_install("numpy")
try:
    import pandas
except ImportError:
    _check_install("pandas")
    
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Core / RAPTOR imports
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.raptor import RaptorProcessor
from rag_lib.chunkers.semantic import SemanticChunker # Used for initial split before RAPTOR
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store

def main():
    setup_environment()
    
    # 1. Configuration
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    # Using a smaller PDF for faster demo if possible, else the main one
    PDF_FILE = DOCS_DIR / "Georgy Volkov ru.pdf" 
    COLLECTION_NAME = "raptor_workflow_demo"

    print_section("1. Loading PDF Data")
    
    pdf_segment = load_pdf_text(PDF_FILE)
    if not pdf_segment:
        print("No PDF loaded. Exiting.")
        return

    segments = [pdf_segment]
    print(f"Loaded raw PDF content (length {len(pdf_segment.content)} chars).")

    # 2. Initial Chunking (for leaf nodes)
    # RAPTOR needs leaf nodes to start with.
    # We can use SemanticChunker or Recursive.
    print_section("2. Initial Chunking (Leaf Nodes)")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Semantic chunking is good for meaningful leaves
    print("Using SemanticChunker...")
    try:
        chunker = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")
        # Semantic chunker expects text, returns documents
        # rag_lib.chunkers.semantic wrapper might be slightly different or same as LangChain
        # Let's see rag_lib implementation: it splits text via split_text
        chunks = chunker.split_text(pdf_segment.content)
        print(f"  -> Generated {len(chunks)} semantic chunks.")
        
        leaf_segments = []
        for i, chunk in enumerate(chunks):
            leaf_segments.append(Segment(
                content=chunk,
                type=SegmentType.TEXT,
                original_format="pdf",
                path=pdf_segment.path,
                segment_id=f"leaf_{i}",
                metadata={"leaf_index": i}
            ))
            
    except Exception as e:
        print(f"Semantic chunking failed/slow ({e}). Fallback to Recursive.")
        from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(pdf_segment.content)
        print(f"  -> Generated {len(chunks)} recursive chunks.")
        
        leaf_segments = []
        for i, chunk in enumerate(chunks):
            leaf_segments.append(Segment(
                content=chunk,
                type=SegmentType.TEXT,
                original_format="pdf",
                path=pdf_segment.path,
                segment_id=f"leaf_{i}",
                metadata={"leaf_index": i}
            ))

    # 3. RAPTOR Processing
    print_section("3. RAPTOR Processing (Hierarchical Summarization)")
    print("Initializing RaptorProcessor...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Using cheaper/faster model for summaries
    
    processor = RaptorProcessor(
        llm=llm,
        embeddings=embeddings,
        max_levels=3
    )
    
    print(f"Processing {len(leaf_segments)} leaf segments... (This may take a while)")
    
    # Process
    try:
        all_segments = processor.process_segments(leaf_segments)
        print(f"RAPTOR processing complete. Total segments: {len(all_segments)} (Original: {len(leaf_segments)})")
        
        # Analyze structure
        levels = {}
        for seg in all_segments:
            lvl = seg.level
            levels[lvl] = levels.get(lvl, 0) + 1
        print("Segment distribution by level:")
        for lvl in sorted(levels.keys()):
            print(f"  Level {lvl}: {levels[lvl]} segments")
            
    except Exception as e:
        print(f"RAPTOR processing failed: {e}")
        # Identify specific missing deps if any
        import traceback
        traceback.print_exc()
        return

    # 4. Indexing
    print_section("4. Indexing Hierarchical Structure")
    
    # Clean previous demo DB
    demo_db_path = "./chroma_demo_db_raptor"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {COLLECTION_NAME}")
    vector_store = get_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    
    print("Indexing segments...")
    indexer.index(all_segments, batch_size=50) # Batch size can be larger
    print("Indexing complete.")

    # 5. Retrieval
    print_section("5. Retrieval (Hierarchical Context)")
    
    # Query relevant to Georgy Volkov (CV)
    query = "What is Georgy Volkov's experience with Python?"
    print(f"Query: '{query}'")
    
    results = vector_store.similarity_search(query, k=3)
    
    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        if res.metadata.get("is_summary_embedding"):
            print("[SUMMARY NODE]")
        else:
            print("[LEAF NODE]")
        print(f"Level: {res.metadata.get('level', '?')}")
        print(f"Content: {res.page_content[:300]}...")
        # Show full metadata to see hierarchy
        # print(f"Metadata: {res.metadata}")

if __name__ == "__main__":
    main()
