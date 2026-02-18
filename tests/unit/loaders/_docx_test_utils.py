from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path
from typing import Optional


_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>
"""


_ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def write_docx(
    tmp_path: Path,
    *,
    document_xml: str,
    styles_xml: Optional[str] = None,
    numbering_xml: Optional[str] = None,
    rels_xml: Optional[str] = None,
    filename: str = "sample.docx",
) -> Path:
    docx_path = tmp_path / filename
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        archive.writestr("_rels/.rels", _ROOT_RELS_XML)
        archive.writestr("word/document.xml", _normalize_xml(document_xml))
        if styles_xml is not None:
            archive.writestr("word/styles.xml", _normalize_xml(styles_xml))
        if numbering_xml is not None:
            archive.writestr("word/numbering.xml", _normalize_xml(numbering_xml))
        if rels_xml is not None:
            archive.writestr("word/_rels/document.xml.rels", _normalize_xml(rels_xml))
    return docx_path


def _normalize_xml(raw: str) -> str:
    return textwrap.dedent(raw).strip()
