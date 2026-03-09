from __future__ import annotations

import re
import textwrap
import zipfile
from pathlib import Path
from typing import Mapping


_DEFAULT_TYPES = {
    "rels": "application/vnd.openxmlformats-package.relationships+xml",
    "xml": "application/xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
}

_OVERRIDE_PATTERNS = [
    (re.compile(r"^ppt/presentation\.xml$"), "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"),
    (re.compile(r"^ppt/slides/slide\d+\.xml$"), "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"),
    (re.compile(r"^ppt/notesSlides/notesSlide\d+\.xml$"), "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"),
    (re.compile(r"^ppt/charts/chart\d+\.xml$"), "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"),
    (re.compile(r"^ppt/diagrams/data\d+\.xml$"), "application/vnd.openxmlformats-officedocument.drawingml.diagramData+xml"),
]

_ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""


def write_pptx(
    tmp_path: Path,
    *,
    parts: Mapping[str, str | bytes],
    filename: str = "sample.pptx",
) -> Path:
    pptx_path = tmp_path / filename
    with zipfile.ZipFile(pptx_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _build_content_types_xml(parts))
        archive.writestr("_rels/.rels", _ROOT_RELS_XML)
        for part_name, content in parts.items():
            archive.writestr(part_name, _normalize_part_content(content))
    return pptx_path


def _build_content_types_xml(parts: Mapping[str, str | bytes]) -> str:
    defaults = "".join(
        f'  <Default Extension="{extension}" ContentType="{content_type}"/>\n'
        for extension, content_type in sorted(_DEFAULT_TYPES.items())
    )
    overrides = []
    for part_name in sorted(parts):
        for pattern, content_type in _OVERRIDE_PATTERNS:
            if pattern.match(part_name):
                overrides.append(
                    f'  <Override PartName="/{part_name}" ContentType="{content_type}"/>\n'
                )
                break

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        f"{defaults}"
        f"{''.join(overrides)}"
        "</Types>\n"
    )


def _normalize_part_content(content: str | bytes) -> str | bytes:
    if isinstance(content, bytes):
        return content
    return textwrap.dedent(content).strip()
