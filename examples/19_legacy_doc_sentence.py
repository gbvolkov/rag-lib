import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, save_json_results, setup_environment

from rag_lib.chunkers.sentence import SentenceSplitter
from rag_lib.loaders.legacy_doc import LegacyDocLoader

"""
E2E Example 19: Legacy DOC Sentence Splitting

Features Tested:
1. LegacyDocLoader: Loading a legacy binary Word `.doc` file as plain text.
2. SentenceSplitter: Splitting the extracted text into sentence-aware segments.
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


def _resolve_doc_path(docs_dir: Path) -> Optional[Path]:
    preferred_names = [
        "Документация.doc",
    ]
    for preferred_name in preferred_names:
        candidate = docs_dir / preferred_name
        if candidate.exists():
            return candidate

    candidates = sorted(path for path in docs_dir.glob("*.doc") if path.is_file())
    if candidates:
        return candidates[0]
    return None


def main() -> None:
    _configure_console_encoding()
    setup_environment()
    print_section("19. Legacy DOC Sentence Splitting")

    docs_dir = Path(__file__).parent.parent / "docs"
    doc_path = _resolve_doc_path(docs_dir)
    if not doc_path:
        print(f"No legacy DOC files found in: {docs_dir}")
        return

    print_section("1. Loading Legacy DOC")
    print(f"Loading {doc_path.name} using LegacyDocLoader...")
    docs = LegacyDocLoader(str(doc_path)).load()

    print(f"Loaded {len(docs)} document(s).")
    if not docs:
        print("No documents loaded from legacy DOC. Exiting.")
        return

    raw_doc = docs[0]
    print(f"Raw text length: {len(raw_doc.page_content)} characters.")
    print(f"Metadata: {raw_doc.metadata}")
    print(f"Preview: {raw_doc.page_content[:240]}...")
    save_json_results(docs, "19_legacy_doc_sentence", "loaded_documents")

    print_section("2. Sentence Splitting")
    splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
        language="russian",
    )
    segments = splitter.split_documents(docs)

    print(f"Generated {len(segments)} segment(s).")
    if not segments:
        print("No sentence segments produced. Exiting.")
        return

    first = segments[0]
    print(f"Sample segment: {first.content[:240]}...")
    print(f"Sample segment metadata: {first.metadata}")
    save_json_results(segments, "19_legacy_doc_sentence", "segments")

    print_section("3. Segment Preview")
    for index, segment in enumerate(segments[:5], start=1):
        print(f"[{index}] {segment.content[:180]}...")
        print(
            "    "
            f"chunk_index={segment.metadata.get('chunk_index', 'n/a')} "
            f"chunk_total={segment.metadata.get('chunk_total', 'n/a')} "
            f"start_index={segment.metadata.get('start_index', 'n/a')}"
        )


if __name__ == "__main__":
    main()
