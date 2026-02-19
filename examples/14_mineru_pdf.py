import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.miner_u import MinerULoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 14: MinerU PDF Workflow

Features Tested:
1. MinerULoader: Layout-aware PDF to markdown document extraction.
2. RecursiveCharacterTextSplitter: Document -> segment chunking.
3. Indexer + VectorStore: Segment indexing via rag_lib abstractions.
4. Vector Retriever: Semantic search over MinerU chunks.
"""


def main() -> None:
    setup_environment()
    print_section("14. MinerU PDF Workflow")

    pdf_path = Path(__file__).parent.parent / "docs" / "statement.pdf"
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}. Please add statement.pdf to docs/.")
        return

    print_section("1. Loading")
    print(f"Loading {pdf_path.name} using MinerULoader...")
    mineru_parse_mode = "txt"
    mineru_timeout_seconds = 1200
    mineru_start_page = int(os.getenv("MINERU_START_PAGE", "0"))
    mineru_end_page = int(os.getenv("MINERU_END_PAGE", "4"))
    print(f"parse_mode: {mineru_parse_mode}")
    print(f"timeout_seconds: {mineru_timeout_seconds}")
    print(f"start_page: {mineru_start_page}")
    print(f"end_page: {mineru_end_page}")
    print("Tip: set MINERU_START_PAGE / MINERU_END_PAGE to widen or narrow the demo range.")
    try:
        loader = MinerULoader(
            str(pdf_path),
            parse_mode=mineru_parse_mode,
            timeout_seconds=mineru_timeout_seconds,
            start_page=mineru_start_page,
            end_page=mineru_end_page,
            parse_formula=False,
            parse_table=False,
        )
        docs = loader.load()
    except ImportError as e:
        print(f"MinerU load failed: {e}")
        print("Install extras with: pip install rag-lib[miner_u]")
        return
    except RuntimeError as e:
        print(f"MinerU load failed: {e}")
        print("Hint: this usually means parse/runtime constraints, not missing extras.")
        print("Try smaller page range (e.g. MINERU_END_PAGE=1) or increase timeout_seconds.")
        return
    except Exception as e:
        print(f"Unexpected MinerU error: {e}")
        return

    if not docs:
        print("No documents loaded from MinerU. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Raw markdown length: {len(docs[0].page_content)} characters.")
    print(f"Sample metadata: {docs[0].metadata}")
    save_json_results(docs, "14_mineru_pdf", "loaded_documents")

    print_section("2. Chunking")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
    segments = splitter.split_documents(docs)

    if not segments:
        print("No segments produced after chunking. Exiting.")
        return

    print(f"Generated {len(segments)} segment(s).")
    print(f"Sample segment: {segments[0].content[:180]}...")
    print(f"Sample segment metadata: {segments[0].metadata}")
    save_json_results(segments, "14_mineru_pdf", "segments")

    print_section("3. Indexing")
    model_name = "text-embedding-3-small"
    embeddings = create_embeddings_model(provider="openai", model_name=model_name)
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="14_mineru_pdf",
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)

    print(f"Embedding model_name: {model_name}")
    print(f"Indexing {len(segments)} segment(s) into '14_mineru_pdf'...")
    indexer.index(segments)
    print("Indexing complete.")

    print_section("4. Retrieval")
    query = "statement date"
    top_k = 3
    print(f"Query: {query}")
    print(f"top_k: {top_k}")

    retriever = create_vector_retriever(vector_store=vector_store, top_k=top_k)
    results = retriever.invoke(query)
    save_json_results(results, "14_mineru_pdf", "retrieved_results")

    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results, start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            "    "
            f"segment_id={metadata.get('segment_id', 'n/a')} "
            f"source={metadata.get('source', 'n/a')} "
            f"parser={metadata.get('parser', 'n/a')}"
        )


if __name__ == "__main__":
    main()
