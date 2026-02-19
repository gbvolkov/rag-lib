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
        document = parse_html_document(raw_bytes)
        strip_non_content_nodes(document)

        if self.output_format == "html":
            content = serialize_html_document(document)
        else:
            blocks = extract_structural_blocks(document)
            content = render_blocks_as_markdown(blocks)

        metadata = {
            "source": self.file_path,
            "source_type": "html",
            "output_format": self.output_format,
        }
        return [Document(page_content=content, metadata=metadata)]
