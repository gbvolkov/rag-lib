from __future__ import annotations

from typing import List

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger
from rag_lib.core.ole_parser import cleanup_extracted_text, extract_doc_text


class LegacyDocLoader:
    """
    Strict legacy Word `.doc` -> plain-text loader.
    """

    def __init__(self, file_path: str, *, cleanup_fields: bool = True) -> None:
        self.file_path = file_path
        self.cleanup_fields = cleanup_fields

    def load(self) -> List[Document]:
        logger.info(
            "Loading legacy DOC: %s cleanup_fields=%s",
            self._safe_log_text(self.file_path),
            self.cleanup_fields,
        )

        text = extract_doc_text(self.file_path)
        if self.cleanup_fields:
            text = cleanup_extracted_text(text)

        metadata = {
            "source": self.file_path,
            "source_type": "doc",
            "output_format": "text",
            "cleanup_fields": self.cleanup_fields,
        }
        return [Document(page_content=text, metadata=metadata)]

    @staticmethod
    def _safe_log_text(value: str) -> str:
        return value.encode("ascii", "backslashreplace").decode("ascii")
