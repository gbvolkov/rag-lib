import pytest
import csv
from rag_lib.loaders.data_loaders import TableLoader

@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "data.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "content"])
        writer.writerow(["1", "A", "Apple"])
        writer.writerow(["2", "A", "Ant"])
        writer.writerow(["3", "B", "Bat"])
    return str(p)

def test_csv_row_segment(csv_file):
    loader = TableLoader(csv_file, mode="row")
    segments = loader.load()
    assert len(segments) == 3
    assert "Apple" in segments[0].content
    assert segments[0].metadata["row"] == 1 # 0-indexed or 1? assuming 1 (skipping header)

def test_csv_groupby(csv_file):
    loader = TableLoader(csv_file, mode="group", group_by="category")
    segments = loader.load()
    assert len(segments) == 2 # A and B
    # Segment A
    seg_a = next(s for s in segments if s.metadata["group_key"] == "A")
    assert "Apple" in seg_a.content
    assert "Ant" in seg_a.content
