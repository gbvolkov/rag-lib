import os
from typing import List, Optional

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None


class PyMuPDFLoader:
    """
    Loads a PDF into one formatted Document using PyMuPDF stack.

    output_format:
    - "markdown": uses pymupdf4llm
    - "html": uses PyMuPDF page HTML extraction and concatenates pages
    """

    def __init__(self, file_path: str, output_format: str = "markdown"):
        self.file_path = file_path
        self.output_format = output_format.lower()
        if self.output_format not in {"markdown", "html"}:
            raise ValueError("output_format must be 'markdown' or 'html'")

    def load(self) -> List[Document]:
        logger.info(f"Loading PDF with PyMuPDF: {self.file_path} format={self.output_format}")

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.output_format == "markdown":
            content = self._load_markdown()
            parser = "PyMuPDF4LLM"
        else:
            content = self._load_html()
            parser = "PyMuPDF"

        page_count = self._get_page_count()
        metadata = {
            "source": self.file_path,
            "parser": parser,
            "output_format": self.output_format,
        }
        if page_count is not None:
            metadata["page_count"] = page_count

        return [Document(page_content=content, metadata=metadata)]

    def _load_markdown(self) -> str:
        if pymupdf4llm is None:
            raise ImportError(
                "pymupdf4llm is required for PyMuPDFLoader(output_format='markdown'). "
                "Install with `pip install pymupdf4llm`."
            )

        try:
            markdown = pymupdf4llm.to_markdown(self.file_path)
        except Exception as e:
            logger.error(f"Failed to extract markdown via pymupdf4llm: {e}")
            raise RuntimeError(f"Failed to extract markdown via pymupdf4llm: {e}")

        if not markdown or not markdown.strip():
            raise RuntimeError("PyMuPDF4LLM returned empty markdown content.")

        return markdown.strip()

    def _load_html(self) -> str:
        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PyMuPDFLoader(output_format='html'). "
                "Install with `pip install pymupdf`."
            )

        doc = fitz.open(self.file_path)
        try:
            parts: List[str] = []
            for page_index, page in enumerate(doc):
                html = page.get_text("html")
                if not html or not html.strip():
                    continue
                parts.append(f"<!-- page: {page_index + 1} -->\n{html}")
        finally:
            doc.close()

        content = "\n\n".join(parts).strip()
        if not content:
            raise RuntimeError("PyMuPDF returned empty HTML content.")

        return content

    def _get_page_count(self) -> Optional[int]:
        if fitz is None:
            return None
        try:
            doc = fitz.open(self.file_path)
            try:
                return len(doc)
            finally:
                doc.close()
        except Exception:
            return None
