from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger
from rag_lib.html_processing import (
    extract_structural_blocks,
    parse_html_document,
    render_blocks_as_markdown,
    serialize_html_document,
    strip_non_content_nodes,
)


def render_html_content(
    content: str | bytes,
    output_format: Literal["markdown", "html"] = "markdown",
) -> str:
    """
    Parse and normalize raw HTML content into markdown or cleaned HTML.
    Strict behavior: parsing/rendering errors are raised to the caller.
    """
    if output_format not in {"markdown", "html"}:
        raise ValueError("output_format must be 'markdown' or 'html'")

    document = parse_html_document(content)
    strip_non_content_nodes(document)

    if output_format == "html":
        return serialize_html_document(document)

    blocks = extract_structural_blocks(document)
    return render_blocks_as_markdown(blocks)


class HTMLLoader:
    """
    Loads HTML files and returns one Document in markdown or html mode.
    Strict behavior: parsing/rendering errors are raised to the caller.
    """

    def __init__(self, file_path: str, output_format: Literal["markdown", "html"] = "markdown"):
        if output_format not in {"markdown", "html"}:
            raise ValueError("output_format must be 'markdown' or 'html'")

        self.file_path = file_path
        self.output_format = output_format

    def load(self) -> List[Document]:
        logger.info(f"Loading HTML: {self.file_path} format={self.output_format}")

        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        raw_bytes = path.read_bytes()
        content = render_html_content(raw_bytes, output_format=self.output_format)

        metadata = {
            "source": self.file_path,
            "source_type": "html",
            "output_format": self.output_format,
        }
        return [Document(page_content=content, metadata=metadata)]
