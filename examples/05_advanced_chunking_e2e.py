import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from example_utils import _check_install, setup_environment, print_section

# Ensure dependencies
try:
    import nltk
except ImportError:
    _check_install("nltk")
try:
    import docx
except ImportError:
    _check_install("python-docx", "docx")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Core imports
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store

def main():
    setup_environment()
    
    # 1. Configuration
    DOCS_DIR = Path(__file__).parent.parent / "docs"
    # Using the parameterized tasks docx which likely has structure
    DOCX_FILE = DOCS_DIR / "Параметризованные задачи.docx" 
    COLLECTION_NAME = "advanced_chunking_demo"

    print_section("1. Loading DOCX for Re-Chunking")
    
    if not DOCX_FILE.exists():
        print(f"File {DOCX_FILE} not found!")
        return

    print(f"Loading {DOCX_FILE.name} using StructuredLoader...")
    # StructuredLoader gives us segments based on document structure (headers)
    loader = StructuredLoader(str(DOCX_FILE))
    initial_segments = loader.load()
    
    print(f"Loaded {len(initial_segments)} segments from structure.")
    
    # Context Reconstruction: tailored for re-chunking
    # We'll concatenate content to form a full text blob to demonstrate other splitters
    print("Reconstructing full text for advanced chunking demonstration...")
    full_text = "\n\n".join([s.content for s in initial_segments])
    print(f"Full text length: {len(full_text)} characters.")

    # 2. Sentence Splitting
    print_section("2. Sentence Splitting")
    print("Initializing SentenceSplitter (Russian language)...")
    
    # NLTK download might happen inside
    sentence_splitter = SentenceSplitter(
        chunk_size=500, 
        chunk_overlap=50, 
        language="russian"
    )
    
    sentence_chunks = sentence_splitter.split_text(full_text)
    print(f"-> SentenceSplitter generated {len(sentence_chunks)} chunks.")
    print(f"   Sample chunk: {sentence_chunks[0][:100]}...")

    # 3. Regex Splitting
    print_section("3. Regex Splitting")
    # Let's split by "Задача" (Task) pattern commonly found in this doc
    # Pattern: Split by "Задача <number>" or just "Задача"
    # Note: RegexSplitter splits *by* the pattern (removing it usually unless capture group)
    pattern = r"(Задача \d+)" 
    print(f"Initializing RegexSplitter with pattern: '{pattern}'")
    
    regex_splitter = RegexSplitter(
        pattern=pattern,
        chunk_size=500,
        chunk_overlap=0 # Regex usually distinct sections
    )
    
    regex_chunks = regex_splitter.split_text(full_text)
    print(f"-> RegexSplitter generated {len(regex_chunks)} chunks.")
    if regex_chunks:
        print(f"   Sample chunk: {regex_chunks[1][:100] if len(regex_chunks)>1 else regex_chunks[0][:100]}...")

    # 4. Convert to Segments & Index (Using Sentence Chunks for Demo)
    print_section("4. Indexing Sentence Chunks")
    
    segments_to_index = []
    for i, chunk in enumerate(sentence_chunks):
        segments_to_index.append(Segment(
            content=chunk,
            type=SegmentType.TEXT,
            original_format="docx",
            path=[DOCX_FILE.name],
            segment_id=f"sent_{i}",
            metadata={
                "source": DOCX_FILE.name,
                "splitter": "sentence",
                "chunk_index": i
            }
        ))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Clean previous demo DB
    demo_db_path = "./chroma_demo_db_advanced"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {COLLECTION_NAME}")
    vector_store = get_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    
    print(f"Indexing {len(segments_to_index)} sentence segments...")
    indexer.index(segments_to_index, batch_size=20)
    print("Indexing complete.")

    # 5. Retrieval
    print_section("5. Retrieval")
    
    query = "параметр" # Parameter
    print(f"Query: '{query}'")
    
    results = vector_store.similarity_search(query, k=2)
    
    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] {res.page_content[:150]}...")
        print(f"    Metadata: {res.metadata}")

if __name__ == "__main__":
    main()
