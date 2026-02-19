import json

from rag_lib.loaders.data_loaders import JsonLoader, QALoader, TableLoader


def test_complex_csv(tmp_path):
    p = tmp_path / "complex.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write('id,notes,"quoted text"\n')
        f.write('1,clean,"simple"\n')
        f.write('2,comma,inside,"quoted, comma"\n')
        f.write('3,newline,"line1\nline2"\n')

    loader = TableLoader(str(p))
    docs = loader.load()

    assert len(docs) == 1
    assert "quoted, comma" in docs[0].page_content
    assert "line1\nline2" in docs[0].page_content


def test_deep_json(tmp_path):
    data = {
        "meta": {"version": 1},
        "response": {
            "items": [
                {"id": "A", "val": 10},
                {"id": "B", "val": 20, "details": {"active": True}},
            ]
        },
    }
    p = tmp_path / "deep.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    loader = JsonLoader(str(p), schema="response.items")
    docs = loader.load()

    assert len(docs) == 1
    assert '"id": "A"' in docs[0].page_content
    assert '"active": true' in docs[0].page_content
    assert docs[0].metadata["schema"] == "response.items"


def test_noisy_qa(tmp_path):
    content = """
    PREAMBLE: This is a FAQ document.
    
    Q: Question 1?
    A: Answer 1.
    
    Random noise or formatting line.
    
    Q: Question 2?
    A: Answer 2
    with multi-line support.
    """
    p = tmp_path / "noise.txt"
    p.write_text(content, encoding="utf-8")

    loader = QALoader(str(p))
    docs = loader.load()

    assert len(docs) == 1
    assert "Question 1?" in docs[0].page_content
    assert "Question 2?" in docs[0].page_content


def test_qa_csv(tmp_path):
    p = tmp_path / "qa.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write('id,question,answer\n')
        f.write('101,"What is A?","A is Alpha."\n')

    loader = TableLoader(str(p))
    docs = loader.load()

    assert len(docs) == 1
    assert "question" in docs[0].page_content
    assert "What is A?" in docs[0].page_content
    assert "A is Alpha." in docs[0].page_content


def test_qa_json(tmp_path):
    data = [
        {"q": "How do I start?", "a": "Press start."},
        {"q": "How to stop?", "a": "Press stop."},
    ]
    p = tmp_path / "qa.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    loader = JsonLoader(str(p), schema=".")
    docs = loader.load()

    assert len(docs) == 1
    assert '"q": "How do I start?"' in docs[0].page_content
    assert '"a": "Press start."' in docs[0].page_content
