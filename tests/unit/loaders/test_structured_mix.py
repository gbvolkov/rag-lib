from __future__ import annotations

import re
from pathlib import Path

from rag_lib.loaders.docx import DocXLoader
from _docx_test_utils import write_docx


def test_docx_loader_mix_headings_and_body(tmp_path: Path) -> None:
    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Chapter 1</w:t></w:r></w:p>
        <w:p><w:r><w:t>Intro text.</w:t></w:r></w:p>
        <w:p><w:r><w:rPr><w:b/></w:rPr><w:t>1.2 Rule</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    markdown = docs[0].page_content
    assert "# Chapter 1" in markdown
    assert "Intro text." in markdown
    assert re.search(r"^###\s+1\.2 Rule$", markdown, flags=re.MULTILINE)


def test_docx_loader_ordered_list_counter(tmp_path: Path) -> None:
    numbering_xml = """
    <w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:abstractNum w:abstractNumId="1">
        <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>
      </w:abstractNum>
      <w:num w:numId="11"><w:abstractNumId w:val="1"/></w:num>
    </w:numbering>
    """

    document_xml = """
    <w:document
      xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <w:body>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="11"/></w:numPr></w:pPr><w:r><w:t>First</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="11"/></w:numPr></w:pPr><w:r><w:t>Second</w:t></w:r></w:p>
      </w:body>
    </w:document>
    """

    docx_path = write_docx(tmp_path, document_xml=document_xml, numbering_xml=numbering_xml)
    docs = DocXLoader(str(docx_path)).load()

    assert len(docs) == 1
    markdown = docs[0].page_content
    assert re.search(r"^1\.\s+First$", markdown, flags=re.MULTILINE)
    assert re.search(r"^2\.\s+Second$", markdown, flags=re.MULTILINE)