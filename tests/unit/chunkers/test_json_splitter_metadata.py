import json

from rag_lib.chunkers.json import JsonSplitter


def test_json_splitter_promotes_recursive_leaf_metadata():
    data = [
        {
            "page_content": "Alpha",
            "metadata": {
                "it_system": "WEB:CRM",
                "nested": {"flag": True},
            },
        }
    ]

    splitter = JsonSplitter(schema=".")
    segments = splitter.create_segments(
        json.dumps(data, ensure_ascii=False),
        metadata={"source": "docs/QA_data.json"},
    )

    assert len(segments) == 1
    meta = segments[0].metadata
    assert meta["source"] == "docs/QA_data.json"
    assert meta["json_index"] == 0
    assert meta["json__page_content"] == "Alpha"
    assert meta["json__metadata__it_system"] == "WEB:CRM"
    assert meta["json__metadata__nested__flag"] is True


def test_json_splitter_metadata_truncates_strings_by_default():
    data = [{"text": "a" * 300}]
    splitter = JsonSplitter(schema=".")
    segments = splitter.create_segments(json.dumps(data))

    promoted = segments[0].metadata["json__text"]
    assert len(promoted) == 256
    assert promoted.endswith("...")


def test_json_splitter_metadata_truncation_can_be_disabled():
    long_value = "b" * 300
    data = [{"text": long_value}]
    splitter = JsonSplitter(schema=".", metadata_value_max_len=None)
    segments = splitter.create_segments(json.dumps(data))

    assert segments[0].metadata["json__text"] == long_value


def test_json_splitter_keeps_common_and_promoted_metadata_for_filtering():
    data = {
        "response": {
            "items": [
                {"metadata": {"it_system": "WEB:CRM"}, "question": "How to login?"},
                {"metadata": {"it_system": "ERP"}, "question": "How to export?"},
            ]
        }
    }

    splitter = JsonSplitter(schema="response.items")
    segments = splitter.create_segments(
        json.dumps(data),
        metadata={"source": "qa.json", "source_type": "json"},
    )

    assert len(segments) == 2
    assert segments[0].metadata["source"] == "qa.json"
    assert segments[0].metadata["source_type"] == "json"
    assert segments[0].metadata["json_index"] == 0
    assert segments[0].metadata["json__metadata__it_system"] == "WEB:CRM"
    assert segments[1].metadata["json_index"] == 1
    assert segments[1].metadata["json__metadata__it_system"] == "ERP"


def test_json_splitter_invalid_schema_path_returns_no_segments():
    data = {"response": {"items": [{"id": "A"}]}}

    splitter = JsonSplitter(schema="response.missing.items")
    segments = splitter.create_segments(json.dumps(data))

    assert segments == []
