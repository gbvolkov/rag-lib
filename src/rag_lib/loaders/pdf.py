import uuid
from typing import List, Optional, Any, Dict
from rag_lib.core.domain import Segment, SegmentType

# Optional import
try:
    import camelot
except ImportError:
    camelot = None

from rag_lib.config import Settings
from rag_lib.summarizers.table import TableSummarizer
from rag_lib.core.logger import logger

class PDFLoader:
    """
    Loads PDF files, specializing in high-fidelity table extraction using Camelot.
    """
    def __init__(self, file_path: str, summarizer: Optional[TableSummarizer] = None, backend: Optional[str] = None):
        self.file_path = file_path
        self.summarizer = summarizer
        
        # Determine backend
        if backend is None:
            backend = Settings().ingestion.default_pdf_backend
        self.backend = backend

        if camelot is None:
            logger.error("camelot-py not found.")
            raise ImportError("camelot-py is required for PDFLoader. Install with `pip install camelot-py[cv]`")

    def load(self) -> List[Segment]:
        logger.info(f"Loading PDF: {self.file_path} using backend={self.backend}")
        segments: List[Segment] = []
        
        # Extract Tables using Camelot
        try:
            tables = camelot.read_pdf(
                self.file_path, 
                pages='all', 
                backend=self.backend 
            )
            logger.debug(f"Found {len(tables)} tables in {self.file_path}")
        except Exception as e:
            # Enhanced error message for common Poppler missing error
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
            # table.df is a pandas DataFrame
            df = table.df
            
            # Convert to Markdown
            # Pandas to_markdown requires 'tabulate'
            try:
                markdown = df.to_markdown(index=False)
            except ImportError:
                 # Fallback manual conversion if tabulate missing
                 markdown = df.to_csv(sep="|") # Crude fallback, but tabulate is standard
            
            metadata={
                "source": self.file_path,
                "page": table.page,
                "table_index": i
            }

            # Generate Summary if summarizer is available
            if self.summarizer:
                try:
                    metadata["summary"] = self.summarizer.summarize(markdown)
                except Exception as e:
                     # Don't fail the whole load if summary fails
                     metadata["summary_error"] = str(e)

            seg = Segment(
                content=markdown,
                segment_id=str(uuid.uuid4()),
                type=SegmentType.TABLE,
                original_format="markdown",
                metadata=metadata
            )
            segments.append(seg)
            
        return segments
