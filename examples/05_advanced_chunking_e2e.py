import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from example_utils import _check_install, print_section, setup_environment

# Ensure dependencies
try:
    import nltk
except ImportError:
    _check_install("nltk")
try:
    import docx
except ImportError:
    _check_install("python-docx", "docx")

from langchain_openai import OpenAIEmbeddings

from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.core.indexer import Indexer
from rag_lib.loaders.docx import DocXLoader
from rag_lib.vectors.factory import create_vector_store


def main() -> None:
    setup_environment()

    # 1. Configuration
    docs_dir = Path(__file__).parent.parent / "docs"
    docx_file = docs_dir / "Параметризованные задачи.docx"
    collection_name = "advanced_chunking_demo"

    print_section("1. Loading DOCX for Re-Chunking")

    if not docx_file.exists():
        print(f"File {docx_file} not found!")
        return

    print(f"Loading {docx_file.name} using DocXLoader...")
    loader = DocXLoader(str(docx_file))
    docs = loader.load()

    print(f"Loaded {len(docs)} markdown document(s).")
    if not docs:
        return

    print("Reconstructing full text for advanced chunking demonstration...")
    full_text = docs[0].page_content
    print(f"Full text length: {len(full_text)} characters.")

    # 2. Sentence Splitting
    print_section("2. Sentence Splitting")
    print("Initializing SentenceSplitter (Russian language)...")

    sentence_splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
        language="russian",
    )

    sentence_chunks = sentence_splitter.split_text(full_text)
    print(f"-> SentenceSplitter generated {len(sentence_chunks)} chunks.")
    if sentence_chunks:
        print(f"   Sample chunk: {sentence_chunks[0][:100]}...")

    # 3. Regex Splitting
    print_section("3. Regex Splitting")
    pattern = r"(Задача \d+)"
    print(f"Initializing RegexSplitter with pattern: '{pattern}'")

    regex_splitter = RegexSplitter(
        pattern=pattern,
        chunk_size=500,
        chunk_overlap=0,
    )

    regex_chunks = regex_splitter.split_text(full_text)
    print(f"-> RegexSplitter generated {len(regex_chunks)} chunks.")
    if regex_chunks:
        sample = regex_chunks[1] if len(regex_chunks) > 1 else regex_chunks[0]
        print(f"   Sample chunk: {sample[:100]}...")

    # 4. Convert to Segments & Index (Using Sentence Chunks for Demo)
    print_section("4. Indexing Sentence Chunks")

    segments_to_index = []
    for i, chunk in enumerate(sentence_chunks):
        segments_to_index.append(
            Segment(
                content=chunk,
                type=SegmentType.TEXT,
                original_format="docx",
                path=[docx_file.name],
                segment_id=f"sent_{i}",
                metadata={
                    "source": docx_file.name,
                    "splitter": "sentence",
                    "chunk_index": i,
                },
            )
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    demo_db_path = "./chroma_demo_db_advanced"
    if os.path.exists(demo_db_path):
        shutil.rmtree(demo_db_path)

    print(f"Initializing Vector Store: {collection_name}")
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
    )

    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Indexing {len(segments_to_index)} sentence segments...")
    indexer.index(segments_to_index, batch_size=20)
    print("Indexing complete.")

    # 5. Retrieval
    print_section("5. Retrieval")

    query = "параметр"
    print(f"Query: '{query}'")

    results = vector_store.similarity_search(query, k=2)

    print(f"\nTop {len(results)} Results:")
    for i, res in enumerate(results):
        print(f"\n[{i + 1}] {res.page_content[:150]}...")
        print(f"    Metadata: {res.metadata}")


if __name__ == "__main__":
    main()