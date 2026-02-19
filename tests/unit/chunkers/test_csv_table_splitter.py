from rag_lib.chunkers.csv_table import CSVTableSplitter
from rag_lib.summarizers.table import TableSummarizer


class MockSummarizer(TableSummarizer):
    def summarize(self, markdown_table: str) -> str:
        row_count = markdown_table.count("\n")
        return f"summary(rows={row_count})"


def test_csv_table_splitter_rows_per_chunk():
    text = "id,name\n1,A\n2,B\n3,C"
    splitter = CSVTableSplitter(max_rows_per_chunk=2, delimiter=",")

    segments = splitter.create_segments(text)

    assert len(segments) == 2
    assert segments[0].metadata["data_row_start"] == 0
    assert segments[0].metadata["data_row_end"] == 2
    assert segments[0].metadata["data_row_count"] == 2
    assert segments[0].metadata["table_chunk_total"] == 2
    assert segments[1].metadata["data_row_start"] == 2
    assert segments[1].metadata["data_row_end"] == 3
    assert segments[1].metadata["data_row_count"] == 1
    assert segments[0].content.splitlines()[0] == "id,name"
    assert segments[1].content.splitlines()[0] == "id,name"


def test_csv_table_splitter_allows_single_row_overflow():
    text = "id,desc\n1,xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    splitter = CSVTableSplitter(max_rows_per_chunk=10, max_chunk_size=8, delimiter=",")

    segments = splitter.create_segments(text)

    assert len(segments) == 1
    assert segments[0].metadata["data_row_count"] == 1
    assert segments[0].metadata["data_row_start"] == 0
    assert segments[0].metadata["data_row_end"] == 1


def test_csv_table_splitter_header_only_produces_no_chunks():
    text = "id,name\n"
    splitter = CSVTableSplitter(max_rows_per_chunk=2, delimiter=",")

    segments = splitter.create_segments(text)

    assert segments == []


def test_csv_table_splitter_summary_metadata_and_content_injection():
    text = "id,name\n1,A\n2,B"
    splitter = CSVTableSplitter(
        max_rows_per_chunk=1,
        delimiter=",",
        summarizer=MockSummarizer(),
        summarize_table=True,
        summarize_chunks=True,
        inject_summaries_into_content=True,
    )

    segments = splitter.create_segments(text)

    assert len(segments) == 2
    assert "table_summary" in segments[0].metadata
    assert "chunk_summary" in segments[0].metadata
    assert "original_content" in segments[0].metadata
    assert segments[0].metadata["table_summary"] == segments[1].metadata["table_summary"]
    assert segments[0].content.startswith("Table Summary:\n")
    assert "\n\n---\n\n" in segments[0].content
