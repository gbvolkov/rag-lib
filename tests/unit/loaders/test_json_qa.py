import json

import pytest

from rag_lib.loaders.data_loaders import JsonLoader, TextLoader


@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "data.json"
    data = [{"id": 1, "text": "foo"}, {"id": 2, "text": "bar"}]
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_json_loader_root_returns_single_document(json_file):
    loader = JsonLoader(json_file, schema=".")
    docs = loader.load()
    assert len(docs) == 1
    assert '"text": "foo"' in docs[0].page_content
    assert '"text": "bar"' in docs[0].page_content
    assert docs[0].metadata["schema"] == "."


def test_json_loader_schema_selection(json_file):
    loader = JsonLoader(json_file, schema="0.text")
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].page_content == '"foo"'


@pytest.fixture
def qa_file(tmp_path):
    content = "Q: What is X?\nA: X is Y.\n\nQ: Who is Z?\nA: Z is generic."
    p = tmp_path / "faq.txt"
    p.write_text(content, encoding="utf-8")
    return str(p)


def test_text_loader_returns_single_document(qa_file):
    loader = TextLoader(qa_file)
    docs = loader.load()
    assert len(docs) == 1
    assert "Q: What is X?" in docs[0].page_content
    assert "Q: Who is Z?" in docs[0].page_content
