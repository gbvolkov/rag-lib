from pathlib import Path
from uuid import uuid4

import pytest

from rag_lib.core.ole_parser import (
    FIELD_BEGIN,
    FIELD_END,
    FIELD_SEPARATOR,
    NotOleWordDocumentError,
    cleanup_extracted_text,
    extract_doc_text,
)


def test_extract_doc_text_rejects_non_ole_file() -> None:
    path = Path.cwd() / f"test-not-a-doc-{uuid4().hex}.doc"
    try:
        path.write_bytes(b"plain text")

        with pytest.raises(NotOleWordDocumentError):
            extract_doc_text(path)
    finally:
        path.unlink(missing_ok=True)


def test_cleanup_extracted_text_keeps_field_result_only() -> None:
    raw = (
        "Email: "
        f"{FIELD_BEGIN} HYPERLINK \"mailto:test@example.com\" "
        f"{FIELD_SEPARATOR}test@example.com{FIELD_END}"
        " done\x07\x01"
    )

    assert cleanup_extracted_text(raw) == "Email: test@example.com done\t"
