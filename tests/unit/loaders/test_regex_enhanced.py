import pytest
from rag_lib.loaders.regex import RegexHierarchyLoader

@pytest.fixture
def temp_text_file(tmp_path):
    def _create(content):
        p = tmp_path / "test_complex.txt"
        p.write_text(content, encoding="utf-8")
        return str(p)
    return _create

# R-05: Deeply Nested Hierarchy
def test_deep_nesting(temp_text_file):
    content = """
# Book 1
Intro to Book 1
## Part A
Part A details
### Chapter 1
#### Section 1.1
Deep content here
## Part B
Part B content
# Book 2
"""
    path = temp_text_file(content)
    loader = RegexHierarchyLoader(path, patterns=[
        (1, r"^# "), (2, r"^## "), (3, r"^### "), (4, r"^#### ")
    ])
    segments = loader.load()
    
    # Verify deep leaf
    deep_seg = next(s for s in segments if "Deep content" in s.content)
    assert deep_seg.level == 4
    assert deep_seg.path == ["Book 1", "Part A", "Chapter 1"]
    
    # Verify reset at Book 2
    book2 = segments[-1]
    assert book2.content.strip() == "# Book 2"
    assert book2.path == [] # Root level reset

# R-06: Mixed Indentation and Spacing
def test_messy_input(temp_text_file):
    content = """
    #  Chapter 1   
    Content with leading spaces.
      ##   Section 1
    Content indented.
    """
    path = temp_text_file(content)
    # Patterns allow flexible whitespace
    loader = RegexHierarchyLoader(path, patterns=[
        (1, r"^\s*#\s+"), (2, r"^\s*##\s+")
    ])
    segments = loader.load()
    
    # Expect 2 segments: 
    # 0. Chapter 1 (Root preamble filtered)
    # 1. Section 1
    assert len(segments) == 2
    assert "Chapter 1" in segments[0].content
    assert "Section 1" in segments[1].content
    assert segments[1].path == ["Chapter 1"]
