import pandas as pd

from rag_lib.loaders.csv_excel import CSVLoader


def test_auto_detect_semicolon(tmp_path):
    csv_file = tmp_path / "semi.csv"
    with open(csv_file, "w", encoding="utf-8") as handle:
        handle.write("col1;col2\n1;a\n2;b")

    loader = CSVLoader(str(csv_file))
    docs = loader.load()

    assert len(docs) == 1
    assert "col1" in docs[0].page_content
    assert "col2" in docs[0].page_content
    assert docs[0].metadata["delimiter"] == ";"
    assert docs[0].metadata["output_format"] == "markdown"


def test_csv_output_format_is_normalized_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame({"col": [1, 2], "txt": ["a", "b"]})
    df.to_csv(csv_file, index=False)

    loader = CSVLoader(str(csv_file), output_format="csv")
    docs = loader.load()

    assert len(docs) == 1
    lines = docs[0].page_content.splitlines()
    assert lines[0] == "col,txt"
    assert lines[1] == "1,a"
    assert lines[2] == "2,b"
    assert docs[0].metadata["output_format"] == "csv"


def test_explicit_delimiter_override(tmp_path):
    csv_file = tmp_path / "tab.csv"
    with open(csv_file, "w", encoding="utf-8") as handle:
        handle.write("h1\th2\nx\ty\n")

    loader = CSVLoader(str(csv_file), output_format="csv", delimiter="\t")
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata["delimiter"] == "\t"
    assert docs[0].page_content.startswith("h1\th2")
