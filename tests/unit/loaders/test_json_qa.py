import pytest
import json
from rag_lib.loaders.data_loaders import JsonLoader, QALoader

@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "data.json"
    data = [{"id": 1, "text": "foo"}, {"id": 2, "text": "bar"}]
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)

def test_json_list_root(json_file):
    loader = JsonLoader(json_file, jq_schema=".")
    segments = loader.load()
    assert len(segments) == 2
    assert "foo" in segments[0].content

@pytest.fixture
def qa_file(tmp_path):
    content = "Q: What is X?\nA: X is Y.\n\nQ: Who is Z?\nA: Z is generic."
    p = tmp_path / "faq.txt"
    p.write_text(content, encoding="utf-8")
    return str(p)

def test_qa_format(qa_file):
    loader = QALoader(qa_file)
    segments = loader.load()
    assert len(segments) == 2
    assert segments[0].metadata["question"] == "What is X?"
    assert segments[0].content == "Q: What is X?\nA: X is Y."
