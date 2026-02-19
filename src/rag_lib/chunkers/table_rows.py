import csv
import io
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class ParsedTable:
    header: List[str]
    rows: List[List[str]]


@dataclass(frozen=True)
class TableRowChunk:
    rows: List[List[str]]
    data_row_start: int
    data_row_end: int


def detect_csv_delimiter(text: str, fallback: str = ",") -> str:
    sample = text[:2048]
    if not sample.strip():
        return fallback
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return fallback


def parse_csv_table(
    text: str,
    delimiter: Optional[str] = None,
    *,
    use_first_row_as_header: bool = True,
) -> tuple[ParsedTable, str]:
    resolved_delimiter = delimiter or detect_csv_delimiter(text)
    if not text.strip():
        return ParsedTable(header=[], rows=[]), resolved_delimiter

    reader = csv.reader(io.StringIO(text), delimiter=resolved_delimiter)
    rows = list(reader)
    if not rows:
        return ParsedTable(header=[], rows=[]), resolved_delimiter

    if use_first_row_as_header:
        header = rows[0]
        width = len(header)
        normalized_rows = [_normalize_row(row, width) for row in rows[1:]]
    else:
        width = max(len(row) for row in rows)
        if width == 0:
            return ParsedTable(header=[], rows=[]), resolved_delimiter
        header = _generate_column_names(width)
        normalized_rows = [_normalize_row(row, width) for row in rows]
    return ParsedTable(header=header, rows=normalized_rows), resolved_delimiter


def render_csv_table(header: List[str], rows: List[List[str]], delimiter: str = ",") -> str:
    if not header:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().rstrip("\n")


def parse_markdown_table(text: str, *, use_first_row_as_header: bool = True) -> ParsedTable:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return ParsedTable(header=[], rows=[])

    first_row = _split_markdown_row(lines[0])
    if not first_row:
        return ParsedTable(header=[], rows=[])

    if not _is_markdown_separator_row(lines[1], expected_columns=len(first_row)):
        return ParsedTable(header=[], rows=[])

    rows: List[List[str]] = []
    width = len(first_row)
    for line in lines[2:]:
        if "|" not in line:
            continue
        row = _normalize_row(_split_markdown_row(line), width)
        rows.append(row)

    if not use_first_row_as_header:
        data_rows = [_normalize_row(first_row, width), *rows]
        return ParsedTable(header=_generate_column_names(width), rows=data_rows)

    header = first_row
    return ParsedTable(header=header, rows=rows)


def render_markdown_table(header: List[str], rows: List[List[str]]) -> str:
    if not header:
        return ""
    header_line = "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |"
    separator_line = "| " + " | ".join("---" for _ in header) + " |"

    data_lines = []
    width = len(header)
    for row in rows:
        normalized = _normalize_row(row, width)
        data_lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in normalized) + " |")

    lines = [header_line, separator_line, *data_lines]
    return "\n".join(lines)


def chunk_table_rows(
    header: List[str],
    rows: List[List[str]],
    render_chunk: Callable[[List[str], List[List[str]]], str],
    *,
    max_rows_per_chunk: Optional[int],
    max_chunk_size: Optional[int],
    length_function: Callable[[str], int],
) -> List[TableRowChunk]:
    if not header or not rows:
        return []

    if max_rows_per_chunk is not None and max_rows_per_chunk <= 0:
        raise ValueError("max_rows_per_chunk must be > 0 when provided.")
    if max_chunk_size is not None and max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0 when provided.")

    chunks: List[TableRowChunk] = []
    row_index = 0
    total_rows = len(rows)

    while row_index < total_rows:
        start = row_index
        chunk_rows: List[List[str]] = []

        while row_index < total_rows:
            next_row = rows[row_index]
            candidate_rows = chunk_rows + [next_row]
            candidate_content = render_chunk(header, candidate_rows)

            exceeds_rows = (
                max_rows_per_chunk is not None and len(candidate_rows) > max_rows_per_chunk
            )
            exceeds_size = (
                max_chunk_size is not None and length_function(candidate_content) > max_chunk_size
            )

            if not chunk_rows:
                # Always keep at least one row with header in the chunk, even if oversized.
                chunk_rows.append(next_row)
                row_index += 1
                if exceeds_rows or exceeds_size:
                    break
                continue

            if exceeds_rows or exceeds_size:
                break

            chunk_rows.append(next_row)
            row_index += 1

        chunks.append(
            TableRowChunk(
                rows=chunk_rows,
                data_row_start=start,
                data_row_end=start + len(chunk_rows),
            )
        )

    return chunks


def build_summary_content(
    original_content: str,
    *,
    table_summary: Optional[str],
    chunk_summary: Optional[str],
) -> str:
    blocks: List[str] = []
    if table_summary and table_summary.strip():
        blocks.append(f"Table Summary:\n{table_summary.strip()}")
    if chunk_summary and chunk_summary.strip():
        blocks.append(f"Chunk Summary:\n{chunk_summary.strip()}")
    if not blocks:
        return original_content
    blocks.append("---")
    blocks.append(original_content)
    return "\n\n".join(blocks)


def _normalize_row(row: List[str], width: int) -> List[str]:
    if len(row) == width:
        return row
    if len(row) > width:
        return row[:width]
    return row + [""] * (width - len(row))


def _split_markdown_row(line: str) -> List[str]:
    working = line.strip()
    if working.startswith("|"):
        working = working[1:]
    if working.endswith("|"):
        working = working[:-1]
    if not working:
        return []
    return [cell.strip() for cell in working.split("|")]


def _is_markdown_separator_row(line: str, expected_columns: int) -> bool:
    cells = _split_markdown_row(line)
    if len(cells) != expected_columns:
        return False
    for cell in cells:
        if not cell:
            return False
        if any(ch not in "-:" for ch in cell):
            return False
        if "-" not in cell:
            return False
    return True


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _generate_column_names(width: int) -> List[str]:
    return [f"Column{i}" for i in range(1, width + 1)]
