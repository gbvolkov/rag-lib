import re
import uuid
from collections import Counter
from typing import List, Optional, Any, Dict, Set
from rag_lib.core.domain import Document
from rag_lib.config import Settings
from rag_lib.summarizers.table import TableSummarizer
from rag_lib.core.logger import logger

# Optional imports
try:
    import camelot
except ImportError:
    camelot = None

try:
    import pypdf
except ImportError:
    pypdf = None

class PDFLoader:
    """
    Loads PDF files.
    Mode 'text' (default): Extracts text using pypdf. Returns 1 Document per file.
    Mode 'table': Extracts tables using Camelot. Returns 1 Document per Table.
    """
    def __init__(self, file_path: str, mode: str = "text", summarizer: Optional[TableSummarizer] = None, backend: Optional[str] = None):
        self.file_path = file_path
        self.mode = mode
        self.summarizer = summarizer
        
        # Backend for Camelot
        if backend is None:
            backend = Settings().ingestion.default_pdf_backend
        self.backend = backend

    def load(self) -> List[Document]:
        logger.info(f"Loading PDF: {self.file_path} mode={self.mode}")
        
        if self.mode == "text":
            return self._load_text()
        elif self.mode == "table":
            return self._load_tables()
        else:
            raise ValueError(f"Unknown mode {self.mode}. Use 'text' or 'table'.")

    def _load_text(self) -> List[Document]:
        if pypdf is None:
             raise ImportError("pypdf is required for PDFLoader(mode='text'). Install `pip install pypdf`.")
        
        try:
            reader = pypdf.PdfReader(self.file_path)
            raw_pages: List[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_pages.append(page_text)

            header_footer_signatures = self._detect_repeated_header_footer_signatures(raw_pages)
            cleaned_pages: List[str] = []
            for page_text in raw_pages:
                cleaned = self._clean_page_text(page_text, header_footer_signatures)
                if cleaned:
                    cleaned_pages.append(cleaned)

            full_text = "\n\n".join(cleaned_pages)
            
            metadata = {
                "source": self.file_path,
                "page_count": len(reader.pages)
            }
            
            return [Document(page_content=full_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Failed to load PDF text: {e}")
            raise RuntimeError(f"Failed to load PDF text: {e}")

    def _clean_page_text(self, text: str, repeated_signatures: Set[str]) -> str:
        """
        Remove common PDF extraction noise in a conservative way:
        - repeated headers/footers detected across pages
        - standalone page markers (e.g., "35", "12 / 92")
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "").replace("\xa0", " ")
        lines = [line.strip() for line in normalized.split("\n")]

        kept: List[str] = []
        for line in lines:
            if not line:
                continue

            if self._is_page_marker_line(line):
                continue

            if self._line_signature(line) in repeated_signatures:
                continue

            kept.append(line)

        # Keep paragraph boundaries, but collapse excessive blank lines.
        page_text = "\n".join(kept)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
        return page_text.strip()

    def _detect_repeated_header_footer_signatures(self, pages: List[str]) -> Set[str]:
        """
        Detect lines that frequently appear in top/bottom page zones.
        These are likely headers/footers and should be removed.
        """
        if len(pages) < 2:
            return set()

        signature_counts: Counter[str] = Counter()
        pages_with_signature: Dict[str, Set[int]] = {}

        for page_index, page_text in enumerate(pages):
            normalized = page_text.replace("\r\n", "\n").replace("\r", "\n")
            lines = [line.strip() for line in normalized.split("\n") if line.strip()]
            if not lines:
                continue

            # Consider only the first/last few lines as header/footer candidates.
            candidates = lines[:4] + lines[-4:]
            for line in candidates:
                sig = self._line_signature(line)
                if not sig:
                    continue
                signature_counts[sig] += 1
                pages_with_signature.setdefault(sig, set()).add(page_index)

        repeated: Set[str] = set()
        total_pages = len(pages)

        for sig, count in signature_counts.items():
            # Remove only lines seen on multiple pages and with broad coverage.
            page_coverage = len(pages_with_signature.get(sig, set()))
            if count >= 2 and page_coverage >= 2 and (page_coverage / total_pages) >= 0.2:
                repeated.add(sig)

        return repeated

    def _line_signature(self, line: str) -> str:
        """
        Canonical signature for repeated-line detection.
        """
        canonical = re.sub(r"\s+", " ", line.strip().lower())
        canonical = re.sub(r"\d+", "", canonical).strip()
        return canonical

    def _is_page_marker_line(self, line: str) -> bool:
        """
        Detect standalone page numbering artifacts.
        """
        s = line.strip()
        if not s:
            return True

        # Plain page number: "35"
        if re.fullmatch(r"\d{1,4}", s):
            return True

        # Numbering formats: "35/92", "35 / 92", "- 35 -"
        if re.fullmatch(r"-?\s*\d{1,4}\s*[/\\]\s*\d{1,4}\s*-?", s):
            return True
        if re.fullmatch(r"-\s*\d{1,4}\s*-", s):
            return True

        return False

    def _load_tables(self) -> List[Document]:
        if camelot is None:
            raise ImportError("camelot-py is required for PDFLoader(mode='table'). Install with `pip install camelot-py[cv]`")
            
        documents = []
        try:
            tables = camelot.read_pdf(
                self.file_path, 
                pages='all', 
                backend=self.backend 
            )
            logger.debug(f"Found {len(tables)} tables in {self.file_path}")
        except Exception as e:
             # Handle common poppler error
            msg = str(e)
            if ("poppler" in msg.lower() or "not found" in msg.lower()) and self.backend == "poppler":
                error_msg = (
                    f"Camelot extraction failed with backend='poppler': {e}\n"
                    "Ensure 'poppler' is installed and in your PATH.\n"
                    "Alternatively, try setting backend='lattice' if detecting grid tables."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.error(f"Camelot extraction failed: {e}")
            raise RuntimeError(f"Camelot extraction failed: {e}")

        for i, table in enumerate(tables):
            df = table.df
            
            # Convert to Markdown
            try:
                markdown = df.to_markdown(index=False)
            except ImportError:
                markdown = df.to_csv(sep="|")
            
            metadata={
                "source": self.file_path,
                "page": table.page,
                "table_index": i
            }

            if self.summarizer:
                try:
                    metadata["summary"] = self.summarizer.summarize(markdown)
                except Exception as e:
                     metadata["summary_error"] = str(e)

            documents.append(Document(page_content=markdown, metadata=metadata))
            
        return documents
