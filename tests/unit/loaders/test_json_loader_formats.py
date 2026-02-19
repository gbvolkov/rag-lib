import json

from rag_lib.loaders.data_loaders import JsonLoader, SchemaDialect


def test_json_loader_default_json_output_returns_raw_text(tmp_path):
    raw = '{"response":{"items":[{"id":"A","text":"hello"}]}}'
    p = tmp_path / "data.json"
    p.write_text(raw, encoding="utf-8")

    docs = JsonLoader(str(p)).load()

    assert len(docs) == 1
    assert docs[0].page_content == raw
    assert docs[0].metadata["source"] == str(p)
    assert docs[0].metadata["source_type"] == "json"
    assert docs[0].metadata["output_format"] == "json"
    assert docs[0].metadata["schema"] == "."
    assert docs[0].metadata["schema_dialect"] == SchemaDialect.DOT_PATH.value


def test_json_loader_schema_selection_serializes_with_readable_unicode(tmp_path):
    problem_description = "\u0417\u0430\u0439\u0442\u0438 \u0432 \u0442\u0435\u0441\u0442"
    data = {
        "response": {
            "items": [
                {
                    "metadata": {
                        "problem_description": problem_description,
                    }
                }
            ]
        }
    }
    p = tmp_path / "unicode.json"
    p.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")

    docs = JsonLoader(str(p), schema="response.items.0.metadata.problem_description").load()

    assert len(docs) == 1
    assert docs[0].page_content == json.dumps(problem_description, ensure_ascii=False)
    assert "\\u0417" not in docs[0].page_content


def test_json_loader_markdown_output_flattens_items(tmp_path):
    data = {
        "response": {
            "items": [
                {
                    "id": "A",
                    "metadata": {
                        "it_system": "WEB:CRM",
                        "tags": ["x", "y"],
                    },
                }
            ]
        }
    }
    p = tmp_path / "nested.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    docs = JsonLoader(str(p), output_format="markdown", schema="response.items").load()

    assert len(docs) == 1
    content = docs[0].page_content
    assert "## Item 1" in content
    assert "- id: A" in content
    assert "- metadata.it_system: WEB:CRM" in content
    assert "- metadata.tags[0]: x" in content
    assert "- metadata.tags[1]: y" in content
    assert docs[0].metadata["output_format"] == "markdown"


def test_json_loader_dot_path_with_index_and_leading_dot(tmp_path):
    data = {"response": {"items": [{"metadata": {"it_system": "WEB:CRM"}}]}}
    p = tmp_path / "path.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    docs = JsonLoader(
        str(p),
        schema=".response.items.0.metadata.it_system",
        schema_dialect="dot_path",
    ).load()

    assert len(docs) == 1
    assert docs[0].page_content == '"WEB:CRM"'


def test_json_loader_invalid_schema_path_returns_empty_result(tmp_path):
    data = {"response": {"items": [{"id": "A"}]}}
    p = tmp_path / "invalid_path.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    docs = JsonLoader(str(p), schema="response.items.4.missing").load()

    assert docs == []


def test_json_loader_schema_can_resolve_null_value(tmp_path):
    data = {"response": {"items": [{"optional": None}]}}
    p = tmp_path / "null_value.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    docs = JsonLoader(str(p), schema="response.items.0.optional").load()

    assert len(docs) == 1
    assert docs[0].page_content == "null"
