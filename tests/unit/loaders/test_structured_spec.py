from __future__ import annotations

from pathlib import Path

from rag_lib.loaders.docx import DocXLoader
from _docx_test_utils import write_docx


def test_docx_loader_returns_single_markdown_document(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:r><w:t>Hello DOCX</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    assert docs[0].page_content == "Hello DOCX"


def test_docx_loader_metadata_contract(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:r><w:t>Metadata check</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    metadata = docs[0].metadata
    assert metadata["source"] == str(docx_path)
    assert metadata["source_type"] == "docx"
    assert metadata["output_format"] == "markdown"