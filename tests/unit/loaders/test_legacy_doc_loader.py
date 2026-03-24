from pathlib import Path

import pytest

from rag_lib.core.ole_parser import InvalidWordDocumentError
from rag_lib.loaders.legacy_doc import LegacyDocLoader


def test_legacy_doc_loader_returns_single_text_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_lib.loaders.legacy_doc.extract_doc_text", lambda path: "raw text")
    monkeypatch.setattr("rag_lib.loaders.legacy_doc.cleanup_extracted_text", lambda text: "clean text")

    docs = LegacyDocLoader("sample.doc", cleanup_fields=True).load()

    assert len(docs) == 1
    assert docs[0].page_content == "clean text"
    assert docs[0].metadata["source"] == "sample.doc"
    assert docs[0].metadata["source_type"] == "doc"
    assert docs[0].metadata["output_format"] == "text"
    assert docs[0].metadata["cleanup_fields"] is True


def test_legacy_doc_loader_can_return_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rag_lib.loaders.legacy_doc.extract_doc_text", lambda path: "raw text")
    cleanup_called = False

    def _cleanup(text: str) -> str:
        nonlocal cleanup_called
        cleanup_called = True
        return text

    monkeypatch.setattr("rag_lib.loaders.legacy_doc.cleanup_extracted_text", _cleanup)

    docs = LegacyDocLoader("sample.doc", cleanup_fields=False).load()

    assert len(docs) == 1
    assert docs[0].page_content == "raw text"
    assert docs[0].metadata["cleanup_fields"] is False
    assert cleanup_called is False


def test_legacy_doc_loader_bubbles_parser_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(path: str) -> str:
        raise InvalidWordDocumentError("broken")

    monkeypatch.setattr("rag_lib.loaders.legacy_doc.extract_doc_text", _raise)

    with pytest.raises(InvalidWordDocumentError, match="broken"):
        LegacyDocLoader("broken.doc").load()
