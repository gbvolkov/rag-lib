import pytest
from textwrap import dedent
from rag_lib.loaders.regex import RegexHierarchyLoader

@pytest.fixture
def temp_spec_file(tmp_path):
    def _create(content):
        p = tmp_path / "spec_test.txt"
        p.write_text(dedent(content).strip(), encoding="utf-8")
        return str(p)
    return _create

# F-01: Multiple Regex per Level
def test_multiple_regex_per_level(temp_spec_file):
    content = """
    # Chapter 1
    Content A
    Title: Chapter 2
    Content B
    """
# ... (rest of file)
    path = temp_spec_file(content)
    # Level 1 matches "# " OR "Title: "
    loader = RegexHierarchyLoader(path, patterns=[
        {"level": 1, "custom_patterns": [r"^# ", r"^Title: "]}
    ])
    segments = loader.load()
    
    segments = loader.load()
    
    assert len(segments) == 2 # Chapter 1, Chapter 2 (Root filtered if empty)
    assert "Chapter 1" in segments[0].content
    assert "Chapter 2" in segments[1].content

# F-02: Exclusion Patterns
def test_exclusion_patterns(temp_spec_file):
    content = """
    # Real Header
    Content
    # Not A Header (Comment)
    # Also Not A Header
    """
    path = temp_spec_file(content)
    loader = RegexHierarchyLoader(path, 
        patterns=[{"level": 1, "pattern": r"^# "}],
        exclude_patterns=[r"Not A Header"]
    )
    segments = loader.load()
    
    # "Real Header" is a segment.
    # The others should be treated as content of Real Header.
    
    assert len(segments) == 1 # Real Header only (Root filtered)
    real_seg = segments[0]
    assert "Real Header" in real_seg.content
    assert "# Not A Header (Comment)" in real_seg.content

# F-03: Leaf Content Concatenation
def test_leaf_content_concatenation(temp_spec_file):
    content = """
    # Book 1
    Intro text.
    ## Part A
    Part details.
    """
    path = temp_spec_file(content)
    loader = RegexHierarchyLoader(path, patterns=[
        {"level": 1, "pattern": r"^# "},
        {"level": 2, "pattern": r"^## "}
    ], include_parent_content=True)
    segments = loader.load()
    
    # Segments:
    # Segments:
    # 0. Book 1 (Level 1)
    # 1. Part A (Level 2)
    # Root (Preamble) is empty/filtered
    
    part_a = segments[1]
    # Requirement: "leaf segments shall also include content from ALL its parents"
    # Format implied: ParentContent + \n + ChildContent
    
    assert "Book 1" in part_a.content
    # "Intro text" is part of Book 1 content.
    assert "Intro text" in part_a.content
    assert "Part A" in part_a.content

# F-04: Level 0 Consistency
def test_level_0_always(temp_spec_file):
    content = "Preamble\n# Chapter 1"
    path = temp_spec_file(content)
    loader = RegexHierarchyLoader(path, patterns=[{"level": 1, "pattern": r"^# "}])
    segments = loader.load()
    
    assert segments[0].level == 0
    assert "Preamble" in segments[0].content
