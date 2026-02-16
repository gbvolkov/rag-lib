import csv
from typing import List, Optional
import pandas as pd
import io
from rag_lib.core.domain import Document
from rag_lib.summarizers.table import TableSummarizer
from rag_lib.config import Settings
from rag_lib.core.logger import logger

class CSVLoader:
    """
    Loads CSV files as a single Document containing a Markdown Table.
    """
    def __init__(self, file_path: str, summarizer: Optional[TableSummarizer] = None):
        """
        Args:
            file_path: Path to CSV file.
            summarizer: Optional summarizer to enrich metadata.
        """
        self.file_path = file_path
        self.summarizer = summarizer

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
            return ","

    def load(self) -> List[Document]:
        logger.info(f"Loading CSV: {self.file_path}")
        
        # 1. Detect Delimiter
        delimiter = self._detect_delimiter()
        logger.debug(f"Detected delimiter: '{delimiter}'")

        try:
            # Load entire CSV
            # Note: For strict "One Document per File", we load all at once.
            df = pd.read_csv(self.file_path, sep=delimiter)
            
            # Convert to Markdown
            try:
                markdown = df.to_markdown(index=False)
            except Exception:
                markdown = df.to_csv(sep="|", index=False)
            
            metadata = {
                "source": self.file_path,
                "row_count": len(df),
                "delimiter": delimiter
            }

            if self.summarizer:
                try:
                    metadata["summary"] = self.summarizer.summarize(markdown)
                except Exception as e:
                    metadata["summary_error"] = str(e)
            
            return [Document(page_content=markdown, metadata=metadata)]

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise RuntimeError(f"Failed to load CSV: {e}")

class ExcelLoader:
    """
    Loads Excel files, returning one Document per Sheet (Markdown Table).
    """
    def __init__(self, file_path: str, summarizer: Optional[TableSummarizer] = None):
        self.file_path = file_path
        self.summarizer = summarizer

    def load(self) -> List[Document]:
        documents: List[Document] = []
        try:
            xls = pd.ExcelFile(self.file_path)
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name)
                
                # Convert to Markdown
                try:
                    markdown = df.to_markdown(index=False)
                except Exception:
                    markdown = df.to_csv(sep="|", index=False)
                
                metadata = {
                    "source": self.file_path,
                    "sheet_name": sheet_name,
                    "row_count": len(df)
                }

                if self.summarizer:
                    try:
                        metadata["summary"] = self.summarizer.summarize(markdown)
                    except Exception as e:
                        metadata["summary_error"] = str(e)

                documents.append(Document(page_content=markdown, metadata=metadata))
                
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise RuntimeError(f"Failed to load Excel: {e}")
            
        return documents
