import pytest
import os
from rag_lib.loaders.regex import RegexHierarchyLoader

# Fixture to create temporary files
@pytest.fixture
def temp_text_file(tmp_path):
    def _create(content):
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")
        return str(p)
    return _create

# R-01: Multi-level Regex
def test_multi_level_regex(temp_text_file):
    content = "# Ch1\nText1\n## Sec1\nText2"
    path = temp_text_file(content)
    loader = RegexHierarchyLoader(path, patterns=[(1, r"^# "), (2, r"^## ")])
    segments = loader.load()
    
    assert len(segments) == 2
    # Seg 1
    assert segments[0].content.strip() == "# Ch1\nText1"
    assert segments[0].level == 1
    # Seg 2
    assert segments[1].content.strip() == "## Sec1\nText2"
    assert segments[1].level == 2
    assert segments[1].path == ["Ch1"] # Parent name in path

# R-02: Path Persistence
def test_path_persistence(temp_text_file):
    content = "# Ch1\n## Sec1\n### Subsec1\nContent"
    path = temp_text_file(content)
    loader = RegexHierarchyLoader(path, patterns=[(1, r"^# "), (2, r"^## "), (3, r"^### ")])
    segments = loader.load()
    
    leaf = segments[-1]
    assert leaf.content.strip() == "### Subsec1\nContent"
    assert leaf.path == ["Ch1", "Sec1"] 

# R-03: State Reset
def test_regex_state_reset(temp_text_file):
    content = "# Ch1\n## Sec1\n# Ch2\n## Sec2"
    path = temp_text_file(content)
    loader = RegexHierarchyLoader(path, patterns=[(1, r"^# "), (2, r"^## ")])
    segments = loader.load()
    
    sec2 = segments[-1]
    assert sec2.content.strip() == "## Sec2"
    assert sec2.path == ["Ch2"] # Should NOT include Ch1 or Sec1

# R-04: No Match Fallback
def test_regex_no_match_fallback(temp_text_file):
    content = "Just text no headers"
    path = temp_text_file(content)
    loader = RegexHierarchyLoader(path, patterns=[(1, r"^# ")])
    segments = loader.load()
    
    assert len(segments) == 1
    assert segments[0].content == "Just text no headers"
    assert segments[0].level == 0
