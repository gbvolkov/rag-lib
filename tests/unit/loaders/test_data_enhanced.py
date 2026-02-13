import pytest
import csv
import json
from rag_lib.loaders.data_loaders import TableLoader, JsonLoader, QALoader

# T-03: Complex CSV (Dirty Data)
def test_complex_csv(tmp_path):
    p = tmp_path / "complex.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write('id,notes,"quoted text"\n')
        f.write('1,clean,"simple"\n')
        f.write('2,comma,inside,"quoted, comma"\n')
        f.write('3,newline,"line1\nline2"\n') # Multiline field
        
    loader = TableLoader(str(p), mode="row")
    segments = loader.load()
    
    assert len(segments) == 3
    assert "quoted, comma" in segments[1].content
    assert "line1\nline2" in segments[2].content

# J-02: Deeply Nested JSON
def test_deep_json(tmp_path):
    data = {
        "meta": {"version": 1},
        "response": {
            "items": [
                {"id": "A", "val": 10},
                {"id": "B", "val": 20, "details": {"active": True}}
            ]
        }
    }
    p = tmp_path / "deep.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    
    # Target response.items
    loader = JsonLoader(str(p), jq_schema="response.items")
    segments = loader.load()
    
    assert len(segments) == 2
    assert "10" in segments[0].content
    assert "active" in segments[1].content

# Q-02: Noisy QA Text
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
    segments = loader.load()
    
    # Expect 3 segments: Preamble, Q1, Q2
    assert len(segments) == 3
    assert segments[1].metadata["question"] == "Question 1?"
    assert "Answer 2\n    with multi-line support" in segments[2].content

# T-04: QA via CSV
def test_qa_csv(tmp_path):
    p = tmp_path / "qa.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        f.write('id,question,answer\n')
        f.write('101,"What is A?","A is Alpha."\n')
        
    loader = TableLoader(str(p), mode="row")
    segments = loader.load()
    
    assert len(segments) == 1
    content = segments[0].content
    assert "question: What is A?" in content
    assert "answer: A is Alpha." in content

# J-03: QA via JSON
def test_qa_json(tmp_path):
    data = [
        {"q": "How do I start?", "a": "Press start."},
        {"q": "How to stop?", "a": "Press stop."}
    ]
    p = tmp_path / "qa.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    
    loader = JsonLoader(str(p), jq_schema=".")
    segments = loader.load()
    
    assert len(segments) == 2
    assert '"q": "How do I start?"' in segments[0].content
    assert '"a": "Press start."' in segments[0].content
