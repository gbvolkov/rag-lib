import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("Installing missing dependency: langchain-openai")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-openai"])
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    print("Installing missing dependency: langchain-chroma")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-chroma"])
    from langchain_chroma import Chroma

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.loaders.pptx import PPTXLoader
from rag_lib.vectors.factory import create_vector_store
from example_utils import setup_environment, print_section

def main():
    setup_environment()
    
    # 1. Configuration
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    TEXT_FILE = DOCS_DIR / "terms&defs.txt"
    MD_FILE = DOCS_DIR / "quotes.toscrape.com_index.md"
    PPTX_FILE = DOCS_DIR / "Digitme Презентация.pptx"
    
    # Use a persistent directory for this example to see it working, 
    # but clean it up or use a unique name to avoid conflicts?
    # Factory uses settings or ./chroma_db. We'll stick to default but maybe clear it first?
    COLLECTION_NAME = "text_workflow_demo"

    print_section("1. Loading Data")
    
    segments = []

    # Load Text Helper
    def load_text_segment(file_path: Path, format_type: str = "text") -> Segment:
        if not file_path.exists():
            print(f"Warning: File {file_path} not found.")
            return None
        print(f"Loading {file_path.name}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return Segment(
                content=content,
                type=SegmentType.TEXT,
                original_format=format_type,
                path=[file_path.name],
                metadata={"source_file": file_path.name}
            )
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None

    # Load files
    text_seg = load_text_segment(TEXT_FILE, "text")
    if text_seg: segments.append(text_seg)
    
    md_seg = load_text_segment(MD_FILE, "markdown")
    if md_seg: segments.append(md_seg)

    print(f"Checking {PPTX_FILE.name}...")
    if PPTX_FILE.exists():
        print(f"Loading {PPTX_FILE.name} with PPTXLoader...")
        try:
            pptx_docs = PPTXLoader(str(PPTX_FILE)).load()
            if pptx_docs:
                segments.append(
                    Segment(
                        content=pptx_docs[0].page_content,
                        type=SegmentType.TEXT,
                        original_format="markdown",
                        path=[PPTX_FILE.name],
                        metadata={"source_file": PPTX_FILE.name},
                    )
                )
        except Exception as e:
            print(f"Error loading {PPTX_FILE.name}: {e}")
    else:
        print(f"Warning: File {PPTX_FILE} not found.")

    print(f"Total raw segments loaded: {len(segments)}")
    if not segments:
        print("No segments loaded. Exiting.")
        return

    # 2. Chunking
    print_section("2. Chunking")
    # Using larger chunk size for demo clarity
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    chunked_segments = []
    for seg in segments:
        print(f"Splitting {seg.path[0]} (length {len(seg.content)} chars)...")
        # Split
        chunks = splitter.split_text(seg.content)
        print(f"  -> Generated {len(chunks)} chunks.")
        
        # Convert to Segments
        for i, chunk in enumerate(chunks):
            new_seg = Segment(
                content=chunk,
                type=SegmentType.TEXT,
                original_format=seg.original_format,
                path=seg.path,
                parent_id=seg.segment_id,
                segment_id=f"{seg.path[0]}_chunk_{i}",
                metadata={
                    "chunk_index": i,
                    "source": seg.path[0]
                }
            )
            chunked_segments.append(new_seg)

    print(f"Total chunked segments: {len(chunked_segments)}")

    # 3. Enrichment
    print_section("3. Enrichment")
    # Only enrich a small subset if we have many, to avoid long wait/cost
    MAX_ENRICH = 3
    print(f"Enriching first {MAX_ENRICH} chunks using ChatOpenAI (gpt-3.5-turbo)...")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    enricher = SegmentEnricher(llm=llm)
    
    segments_to_enrich = chunked_segments[:MAX_ENRICH]
    remaining_segments = chunked_segments[MAX_ENRICH:]
    
    try:
        enriched_subset = enricher.enrich(segments_to_enrich)
    except Exception as e:
        print(f"Enrichment failed (likely API key or network): {e}")
        enriched_subset = segments_to_enrich
    
    final_segments = enriched_subset + remaining_segments

    # 4. Indexing
    print_section("4. Indexing")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print(f"Initializing Vector Store: {COLLECTION_NAME}")
    # Force a local path for this demo to avoid messing with other DBs
    demo_db_path = "./chroma_demo_db_text"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path) # Start fresh
        
    # We bypass factory slightly to enforce path for demo, 
    # OR we can just use factory and rely on unique collection name.
    # Factory uses settings. let's instantiate Chroma directly to be safe and explicit for demo?
    # Or strict adherence to library patterns. The canonical factory is create_vector_store.
    # Ideally we should use create_vector_store.
    # But factory.py relies on checking settings or defaulting to ./chroma_db.
    # Let's use create_vector_store but be aware it might mix. 
    # Actually, we can just use a unique collection name.
    
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    
    print(f"Indexing {len(final_segments)} segments...")
    indexer.index(final_segments, batch_size=20)
    print("Indexing complete.")

    # 5. Retrieval
    print_section("5. Retrieval")
    
    # Define a query relevant to terms&defs.txt (likely banking/finance terms based on file names)
    # terms&defs.txt likely has "Term: Definition".
    query = "banking guarantee" 
    print(f"Query: '{query}'")
    
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {res.page_content[:300]}...") # Truncate for display
        print(f"Metadata: {res.metadata}")
        if "generated_title" in res.metadata:
            print(f"Enriched Title: {res.metadata['generated_title']}")

if __name__ == "__main__":
    main()
