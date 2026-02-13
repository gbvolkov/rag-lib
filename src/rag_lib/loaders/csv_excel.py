import csv
from typing import List, Optional
import pandas as pd
import uuid
import io
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.summarizers.table import TableSummarizer

from rag_lib.config import Settings
from rag_lib.core.logger import logger

class CSVLoader:
    """
    Loads CSV files as Table Segments.
    Supports auto-detection of delimiters and streaming for large files.
    """
    def __init__(self, file_path: str, summarizer: Optional[TableSummarizer] = None, chunk_size: Optional[int] = None):
        """
        Args:
            file_path: Path to CSV file.
            summarizer: Optional summarizer.
            chunk_size: Number of rows per segment. Defaults to config if None.
        """
        self.file_path = file_path
        self.summarizer = summarizer
        if chunk_size is None:
            chunk_size = Settings().ingestion.chunk_size
        self.chunk_size = chunk_size

    def _detect_delimiter(self, sample_bytes: int = 2048) -> str:
        """
        Auto-detects delimiter using csv.Sniffer.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(sample_bytes)
                if not sample:
                    return ","
                dialect = csv.Sniffer().sniff(sample)
                return dialect.delimiter
        except Exception:
            # Fallback
            return ","

    def load(self) -> List[Segment]:
        logger.info(f"Loading CSV: {self.file_path} with chunk_size={self.chunk_size}")
        segments: List[Segment] = []
        
        # 1. Detect Delimiter (Sniffer)
        delimiter = ","
        try:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                sample = f.read(2048)
                if sample: # Avoid sniffing empty files
                    delimiter = csv.Sniffer().sniff(sample).delimiter
                    logger.debug(f"Detected CSV delimiter: '{delimiter}'")
        except Exception as e:
            logger.warning(f"Could not sniff delimiter for {self.file_path}, defaulting to comma. Error: {e}")
            delimiter = ","
            
        # 2. Read CSV using Pandas with chunks
        try:
            # Use chunks to handle large files
            chunk_iterator = pd.read_csv(
                self.file_path, 
                sep=delimiter,
                chunksize=self.chunk_size
            )
            
            for i, df in enumerate(chunk_iterator):
                # Convert chunk to Markdown
                try:
                    markdown = df.to_markdown(index=False)
                except ImportError:
                     markdown = df.to_csv(sep="|", index=False)
                
                metadata={
                    "source": self.file_path,
                    "row_count": len(df),
                    "chunk_index": i
                }

                if self.summarizer:
                    try:
                        metadata["summary"] = self.summarizer.summarize(markdown)
                    except Exception as e:
                        metadata["summary_error"] = str(e)
                
                seg = Segment(
                    content=markdown,
                    segment_id=str(uuid.uuid4()),
                    type=SegmentType.TABLE,
                    original_format="markdown",
                    metadata=metadata
                )
                segments.append(seg)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")
            
        return segments

class ExcelLoader:
    """
    Loads Excel files (each sheet as a Table Segment).
    """
    def __init__(self, file_path: str, summarizer: Optional[TableSummarizer] = None):
        self.file_path = file_path
        self.summarizer = summarizer

    def load(self) -> List[Segment]:
        segments: List[Segment] = []
        try:
            # Load Excel
            xls = pd.ExcelFile(self.file_path)
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                
                # Convert to Markdown
                try:
                    markdown = df.to_markdown(index=False)
                except ImportError:
                     markdown = df.to_csv(sep="|", index=False)
                
                metadata={
                    "source": self.file_path,
                    "sheet_name": sheet_name,
                    "row_count": len(df)
                }

                if self.summarizer:
                    try:
                        metadata["summary"] = self.summarizer.summarize(markdown)
                    except Exception as e:
                        metadata["summary_error"] = str(e)

                seg = Segment(
                    content=markdown,
                    segment_id=str(uuid.uuid4()),
                    type=SegmentType.TABLE,
                    original_format="markdown",
                    metadata=metadata
                )
                segments.append(seg)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load Excel: {e}")
            
        return segments
