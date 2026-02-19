import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.core.indexer import Indexer
from rag_lib.embeddings.factory import create_embeddings_model
from rag_lib.loaders.docx import DocXLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 06: DOCX Regex Workflow

Features Tested:
1. DocXLoader: Loading DOCX as markdown.
2. RegexSplitter: Splitting text based on regex boundaries.
3. RegexRetriever: Pattern-style retrieval over structured segments.
4. VectorStore + Indexer: Secondary semantic retrieval for comparison.
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


def _resolve_docx_path(docs_dir: Path) -> Optional[Path]:
    preferred_names = [
        "KP_IT_IB_Strategy_Recalc_v7_AppC.docx",
    ]
    for preferred_name in preferred_names:
        preferred_path = docs_dir / preferred_name
        if preferred_path.exists():
            return preferred_path

    candidates = sorted(docs_dir.glob("*.docx"))
    if candidates:
        return candidates[0]
    return None


def _artifact_slug(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    if not slug:
        return "query"
    return slug[:max_len]


def _print_results(label: str, results: list, limit: int = 5) -> None:
    print(f"{label}: {len(results)}")
    for i, doc in enumerate(results[:limit], start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source_file", metadata.get("source", "n/a"))
        print(f"[{i}] {doc.page_content[:180]}...")
        print(
            f"    segment_id={metadata.get('segment_id', 'n/a')} "
            f"chunk_index={metadata.get('chunk_index', 'n/a')} "
            f"source={source}"
        )


def main() -> None:
    _configure_console_encoding()
    setup_environment()
    print_section("06. DOCX Regex Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    docx_path = _resolve_docx_path(docs_dir)
    if not docx_path:
        print(f"No DOCX files found in: {docs_dir}")
        return

    print_section("1. Loading DOCX")
    print(f"Loading {docx_path.name} using DocXLoader...")
    loader = DocXLoader(str(docx_path))
    docs = loader.load()
    if not docs:
        print("No content loaded from DOCX. Exiting.")
        return

    print(f"Loaded {len(docs)} document(s).")
    print(f"Extracted {len(docs[0].page_content)} characters.")
    save_json_results(docs, "06_docx_regex", "loaded_documents")

    print_section("2. Regex Segmentation")
    pattern = (
        r"(?m)(?=^(?:"
        r"#\s+\d+\.\s+.+|"
        r"##\s+\d+\.\d+\.\s+.+|"
        r"###\s+Этап\s+(?:Э\d+|PA)\.\s+.+|"
        r"\*\*(?:D\d+-\d+|PA-\d+)\.\s+.+\*\*|"
        r"-\s+(?:D\d+-\d+|PA-\d+)\.\s+.+|"
        r"\|\s*(?:D\d+-\d+|PA-\d+)\s*\|"
        r"))"
    )

    
    splitter = RegexSplitter(pattern=pattern, chunk_size=1200, chunk_overlap=0)

    segments = splitter.split_documents(docs)
    print(f"Regex splitter produced {len(segments)} segments.")
    if not segments:
        print("No segments produced after regex splitting. Exiting.")
        return

    print(f"Sample segment preview: {segments[0].content[:180]}...")
    print(f"Sample segment metadata: {segments[0].metadata}")
    save_json_results(segments, "06_docx_regex", "segments")

    print_section("3. Regex Retrieval")
    queries = [
        "Состав работ",
        "Команда",
        "трудозатраты",
        "этапы работ",
        "Стоимость",
        "Цель проекта",
    ]
    segment_docs = [(segment.content, segment.to_langchain()) for segment in segments]
    for query in queries:
        query_pattern = re.compile(
            re.sub(r"\\\s+", r"\\s+", re.escape(query.strip())),
            re.IGNORECASE,
        )
        results = [doc for content, doc in segment_docs if query_pattern.search(content)]
        save_json_results(results, "06_docx_regex", f"regex_results_q_{_artifact_slug(query)}")
        print(f"Query '{query}' -> {len(results)} match(es)")
        if results:
            _print_results(f"Regex Results for '{query}'", results)
        else:
            print("No regex matches for this query.")

    print_section("4. Vector Retrieval (Secondary)")
    embeddings = create_embeddings_model(provider="openai", model_name="text-embedding-3-small")
    collection_name = f"06_docx_regex_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vector_store = create_vector_store(
        provider="chroma",
        embeddings=embeddings,
        collection_name=collection_name,
        cleanup=True,
    )
    indexer = Indexer(vector_store=vector_store, embeddings=embeddings)
    indexer.index(segments)
    print(f"Indexed {len(segments)} segments into Chroma collection '{collection_name}'.")

    for vector_query in queries:
        print(f"Vector query: {vector_query}")
        vector_retriever = create_vector_retriever(vector_store=vector_store, top_k=5)
        vector_results = vector_retriever.invoke(vector_query)
        save_json_results(
            vector_results,
            "06_docx_regex",
            f"vector_results_q_{_artifact_slug(vector_query)}",
        )

        if vector_results:
            _print_results("Top Vector Results", vector_results)
        else:
            print("Vector retriever returned no results.")


if __name__ == "__main__":
    main()
