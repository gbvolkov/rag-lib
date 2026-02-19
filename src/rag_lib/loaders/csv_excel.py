import csv
from typing import Any, List, Literal, Optional
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
                "output_format": self.output_format,
            }
            
            return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise RuntimeError(f"Failed to load CSV: {e}")

class ExcelLoader:
    """
    Loads Excel files, returning one Document per sheet.

    Modes:
    - markdown: Preserves mixed sheet content (text blocks + one or more table blocks)
      in a single markdown-oriented document per sheet.
    - csv: Treats each sheet as tabular data and emits normalized CSV text per sheet.
    """

    def __init__(
        self,
        file_path: str,
        output_format: Literal["markdown", "csv"] = "markdown",
        delimiter: str = ",",
        summarizer: Optional[TableSummarizer] = None,
    ):
        if output_format not in ("markdown", "csv"):
            raise ValueError("output_format must be either 'markdown' or 'csv'.")
        self.file_path = file_path
        self.output_format = output_format
        self.delimiter = delimiter
        self.summarizer = summarizer

    def load(self) -> List[Document]:
        documents: List[Document] = []
        try:
            xls = pd.ExcelFile(self.file_path)

            for sheet_name in xls.sheet_names:
                if self.output_format == "csv":
                    content, row_count = self._render_sheet_as_csv(sheet_name=sheet_name)
                else:
                    content, row_count = self._render_sheet_as_markdown(sheet_name=sheet_name)

                metadata = {
                    "source": self.file_path,
                    "sheet_name": sheet_name,
                    "row_count": row_count,
                    "output_format": self.output_format,
                }

                if self.summarizer:
                    try:
                        metadata["summary"] = self.summarizer.summarize(content)
                    except Exception as e:
                        metadata["summary_error"] = str(e)

                documents.append(Document(page_content=content, metadata=metadata))

        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise RuntimeError(f"Failed to load Excel: {e}")

        return documents

    def _render_sheet_as_csv(self, sheet_name: str) -> tuple[str, int]:
        """
        CSV mode: process each sheet as a tabular dataframe.
        """
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        content = df.to_csv(index=False, sep=self.delimiter, lineterminator="\n").rstrip("\n")
        return content, len(df)

    def _render_sheet_as_markdown(self, sheet_name: str) -> tuple[str, int]:
        """
        Markdown mode: preserve mixed content by detecting text/table blocks
        in sheet row order and rendering them into one document per sheet.
        """
        # header=None keeps raw sheet rows so text blocks and multiple tables are retained.
        df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None, dtype=object)
        if df.empty:
            return "", 0

        raw_rows: List[List[str]] = [
            [self._normalize_cell_value(cell) for cell in row]
            for row in df.itertuples(index=False, name=None)
        ]

        row_count = sum(1 for row in raw_rows if any(cell for cell in row))
        blocks = self._split_row_blocks(raw_rows)
        parts: List[str] = []
        for block_kind, block_rows in blocks:
            if block_kind == "text":
                text_part = self._render_text_block(block_rows)
                if text_part:
                    parts.append(text_part)
            else:
                table_part = self._render_markdown_table_block(block_rows)
                if table_part:
                    parts.append(table_part)

        return "\n\n".join(parts).strip(), row_count

    def _split_row_blocks(self, raw_rows: List[List[str]]) -> List[tuple[str, List[List[str]]]]:
        blocks: List[tuple[str, List[List[str]]]] = []
        current_kind: Optional[str] = None
        current_rows: List[List[str]] = []

        for row in raw_rows:
            trimmed = self._trim_trailing_empty_cells(row)
            kind = self._classify_row(trimmed)

            if kind == "blank":
                if current_kind and current_rows:
                    blocks.append((current_kind, current_rows))
                current_kind = None
                current_rows = []
                continue

            if current_kind is None:
                current_kind = kind
                current_rows = [trimmed]
                continue

            if kind != current_kind:
                blocks.append((current_kind, current_rows))
                current_kind = kind
                current_rows = [trimmed]
            else:
                current_rows.append(trimmed)

        if current_kind and current_rows:
            blocks.append((current_kind, current_rows))

        return blocks

    @staticmethod
    def _normalize_cell_value(cell: Any) -> str:
        if cell is None:
            return ""
        # Handles numpy/pandas NA scalars while preserving regular values.
        if pd.isna(cell):
            return ""
        return str(cell).strip()

    @staticmethod
    def _trim_trailing_empty_cells(row: List[str]) -> List[str]:
        trimmed = list(row)
        while trimmed and trimmed[-1] == "":
            trimmed.pop()
        return trimmed

    @staticmethod
    def _classify_row(row: List[str]) -> str:
        if not row:
            return "blank"
        non_empty_cells = sum(1 for cell in row if cell)
        if non_empty_cells == 0:
            return "blank"
        if non_empty_cells == 1:
            return "text"
        return "table"

    @staticmethod
    def _render_text_block(rows: List[List[str]]) -> str:
        lines: List[str] = []
        for row in rows:
            values = [cell for cell in row if cell]
            if values:
                lines.append(" ".join(values))
        return "\n".join(lines).strip()

    def _render_markdown_table_block(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        padded_rows = [row + [""] * (width - len(row)) for row in rows]

        header = [
            cell if cell else f"col_{idx + 1}"
            for idx, cell in enumerate(padded_rows[0])
        ]
        body = padded_rows[1:]

        table_df = pd.DataFrame(body, columns=header) if body else pd.DataFrame(columns=header)
        try:
            return table_df.to_markdown(index=False)
        except Exception:
            return table_df.to_csv(index=False, sep=self.delimiter, lineterminator="\n").rstrip("\n")
