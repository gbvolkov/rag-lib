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
from rag_lib.loaders.legacy_doc import LegacyDocLoader
from rag_lib.retrieval.retrievers import create_vector_retriever
from rag_lib.vectors.factory import create_vector_store

"""
E2E Example 06: Word Regex Workflow

Features Tested:
1. Word loaders: DOCX as markdown or legacy DOC as text.
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


def _resolve_word_path(docs_dir: Path) -> Optional[Path]:
    preferred_names = [
        "Документация.doc",
    ]
    for preferred_name in preferred_names:
        preferred_path = docs_dir / preferred_name
        if preferred_path.exists():
            return preferred_path

    doc_candidates = sorted(docs_dir.glob("*.doc"))
    if doc_candidates:
        return doc_candidates[0]

    docx_candidates = sorted(docs_dir.glob("*.docx"))
    if docx_candidates:
        return docx_candidates[0]
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
    print_section("06. Word Regex Workflow")

    docs_dir = Path(__file__).parent.parent / "docs"
    word_path = _resolve_word_path(docs_dir)
    if not word_path:
        print(f"No Word files found in: {docs_dir}")
        return

    print_section("1. Loading Word")
    if word_path.suffix.lower() == ".doc":
        print(f"Loading {word_path.name} using LegacyDocLoader...")
        loader = LegacyDocLoader(str(word_path))
    else:
        print(f"Loading {word_path.name} using DocXLoader...")
        loader = DocXLoader(str(word_path))

    docs = loader.load()
    if not docs:
        print("No content loaded from Word document. Exiting.")
        return


if __name__ == "__main__":
    main()
