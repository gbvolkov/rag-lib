import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.stores import InMemoryStore

from example_utils import print_section, save_json_results, setup_environment
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.llm.factory import create_llm
from rag_lib.loaders.pptx import PPTXLoader
from rag_lib.retrieval.composition import create_scored_dual_storage_retriever
from rag_lib.retrieval.scored_retriever import SearchType
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 15: PPTX Full Cycle Workflow

Features Tested:
1. PPTXLoader with optional visual summarization enabled.
2. RegexSplitter for slide-level parent segmentation.
3. RecursiveCharacterTextSplitter for shorter child chunks.
4. Dual storage indexing (vector children + doc_store parents).
5. Scored retrieval with hydrated slide-level results.
"""


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass


def _resolve_pptx_path(docs_dir: Path) -> Optional[Path]:
    candidates = sorted(docs_dir.glob("*.pptx"))
    for candidate in candidates:
        if "digitme" in candidate.name.lower():
            return candidate
    if candidates:
        return candidates[0]
    return None


def _artifact_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    if not slug:
        return "query"
    return slug[:max_len]


def _print_results(results: list, limit: int = 5) -> None:
    print(f"Retrieved {len(results)} result(s).")
    for i, doc in enumerate(results[:limit], start=1):
        metadata = doc.metadata or {}
        print(f"[{i}] score={metadata.get('similarity_score', 'n/a')}")
        print(f"    parent_id={metadata.get('parent_id', 'n/a')}")
        print(f"    {doc.page_content[:220]}...")


def main() -> None:
    _configure_console_encoding()
    setup_environment()
    print_section("15. PPTX Full Cycle Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    pptx_path = _resolve_pptx_path(docs_dir)
    if not pptx_path:
        print(f"No PPTX files found in: {docs_dir}")
        return

    print_section("1. Loading PPTX + Visual Summaries")
    print(f"Loading {pptx_path.name} using PPTXLoader(summarize_visuals=True)...")
    llm = create_llm(provider="openai", model_name="gpt-5-mini", streaming=False)
    loader = PPTXLoader(
        str(pptx_path),
        include_notes=True,
        summarize_visuals=True,
        llm=llm,
    )
    docs = loader.load()

    if len(docs) != 1:
        print(f"Unexpected document count: {len(docs)}")
        return

    doc = docs[0]
    print(f"Loaded {len(docs)} document(s).")
    print(f"Markdown length: {len(doc.page_content)} characters.")
    print(f"Metadata: {doc.metadata}")
    print(f"Markdown preview: {doc.page_content[:240]}...")
    save_json_results(docs, "15_pptx_full_cycle", "loaded_documents")

    print_section("2. Slide Segmentation (RegexSplitter)")
    slide_pattern = r"(?m)(?=^# Slide \d+: .+$)"
    logical_splitter = RegexSplitter(pattern=slide_pattern, chunk_size=4000, chunk_overlap=0)
    parent_segments = logical_splitter.split_documents(docs)

    if not parent_segments:
        print("No slide-level parent segments produced. Exiting.")
        return

    print(f"Regex splitter produced {len(parent_segments)} slide parent segment(s).")
    print(f"Sample parent metadata: {parent_segments[0].metadata}")
    print(f"Sample parent content: {parent_segments[0].content[:220]}...")
    save_json_results(parent_segments, "15_pptx_full_cycle", "parent_segments")

    print_section("3. Shorter Child Chunking")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    child_segments = child_splitter.split_segments(parent_segments)

    if not child_segments:
        print("No child chunks produced. Exiting.")
        return

    print(f"Generated {len(child_segments)} child chunk(s).")
    print(f"Sample child parent_id: {child_segments[0].metadata.get('parent_id')}")
    print(f"Sample child metadata: {child_segments[0].metadata}")
    print(f"Sample child content: {child_segments[0].content[:220]}...")
    save_json_results(child_segments, "15_pptx_full_cycle", "child_segments")

    print_section("4. Dual Store Indexing")
    embedding_model_name = "text-embedding-3-small"
    embeddings = create_embeddings_model(provider="openai", model_name=embedding_model_name)
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name="15_pptx_full_cycle",
        cleanup=True,
    )
    doc_store = InMemoryStore()

    indexer = Indexer(vector_store=vector_store, embeddings=embeddings, doc_store=doc_store)
    print(f"Embedding model: {embedding_model_name}")
    print(
        f"Indexing {len(child_segments)} child chunk(s) and "
        f"{len(parent_segments)} parent slide segment(s)..."
    )
    indexer.index(child_segments, parent_segments=parent_segments)
    print("Indexing complete.")

    print_section("5. Retrieval")
    retriever = create_scored_dual_storage_retriever(
        vector_store=vector_store,
        doc_store=doc_store,
        id_key="parent_id",
        search_kwargs={"k": 6},
        search_type=SearchType.similarity_score_threshold,
        score_threshold=0.0,
    )

    queries = [
        "What services does Digitme connect?",
        "What benefits do clients get from the automation?",
        "How does the team work with a client project?",
    ]

    for query in queries:
        print(f"Query: {query}")
        results = retriever.invoke(query)
        save_json_results(
            results,
            "15_pptx_full_cycle",
            f"retrieved_results_q_{_artifact_slug(query)}",
        )
        _print_results(results)


if __name__ == "__main__":
    main()
