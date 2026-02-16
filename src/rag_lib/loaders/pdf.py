import uuid
from typing import List, Optional, Any, Dict
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
            text_content = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            
            full_text = "\n\n".join(text_content)
            
            metadata = {
                "source": self.file_path,
                "page_count": len(reader.pages)
            }
            
            return [Document(page_content=full_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Failed to load PDF text: {e}")
            raise RuntimeError(f"Failed to load PDF text: {e}")

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
