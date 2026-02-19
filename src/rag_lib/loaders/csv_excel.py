import csv
from typing import List, Literal, Optional
import pandas as pd
from rag_lib.core.domain import Document
from rag_lib.summarizers.table import TableSummarizer
from rag_lib.core.logger import logger

class CSVLoader:
    """
    Loads CSV files as a single Document with configurable output format.
    """

    def __init__(
        self,
        file_path: str,
        output_format: Literal["markdown", "csv"] = "markdown",
        delimiter: Optional[str] = None,
    ):
        """
        Args:
            file_path: Path to CSV file.
            output_format: Output representation for table text ('markdown' or 'csv').
            delimiter: Optional explicit CSV delimiter. Auto-detected when omitted.
        """
        if output_format not in ("markdown", "csv"):
            raise ValueError("output_format must be either 'markdown' or 'csv'.")

        self.file_path = file_path
        self.output_format = output_format
        self.delimiter = delimiter

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
        delimiter = self.delimiter or self._detect_delimiter()
        logger.debug(f"Detected delimiter: '{delimiter}'")

        try:
            # Load entire CSV
            # Note: For strict "One Document per File", we load all at once.
            df = pd.read_csv(self.file_path, sep=delimiter)
            
            # Convert to requested output format.
            if self.output_format == "csv":
                content = df.to_csv(index=False, sep=delimiter, lineterminator="\n").rstrip("\n")
            else:
                try:
                    content = df.to_markdown(index=False)
                except Exception:
                    content = df.to_csv(index=False, sep=delimiter, lineterminator="\n").rstrip("\n")
            
            metadata = {
                "source": self.file_path,
                "row_count": len(df),
                "delimiter": delimiter,
                "table_format": self.output_format,
            }
            
            return [Document(page_content=content, metadata=metadata)]

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
