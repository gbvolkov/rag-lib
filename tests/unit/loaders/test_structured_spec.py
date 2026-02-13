import pytest
from unittest.mock import patch
from rag_lib.loaders.structured import StructuredLoader

def create_mock_xml(paragraphs):
    # paragraphs: list of (style, text)
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    for style, text in paragraphs:
        xml += '<w:p>'
        if style:
            xml += f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        xml += f'<w:r><w:t>{text}</w:t></w:r></w:p>'
    xml += '</w:body></w:document>'
    return xml.encode('utf-8')

@pytest.fixture
def mock_loader_factory():
    def _create(paragraphs, **kwargs):
        xml = create_mock_xml(paragraphs)
        with patch('zipfile.ZipFile') as mock_zip:
            instance = mock_zip.return_value.__enter__.return_value
            instance.read.return_value = xml
            loader = StructuredLoader("dummy.docx", **kwargs)
            return loader.load()
    return _create

# F-05: Structured Exclusion
def test_structured_exclusion(mock_loader_factory):
    # H1 "Intro", H1 "Draft", H2 "Details"
    # "Draft" should be excluded (treated as body of Intro? or just ignored as header?)
    # If excluded as header, it becomes body text.
    # So "Intro" -> "Draft" -> "Details"
    # If "Draft" is body, then "Details" is child of "Intro"? 
    # Or "Details" is child of "Draft" (if Draft was H1)? 
    # If Draft is NOT header, it's just text. So level 0 (relative).
    # So "Details" (H2) becomes child of "Intro" (H1).
    
    segments = mock_loader_factory(
        [
            ("Heading1", "Intro"),
            ("Heading1", "Draft Header"),
            ("Heading2", "Details")
        ],
        exclude_patterns=["Draft"]
    )
    
    # Segments:
    # 0. Intro (L1) (Root filtered)
    # 1. Details (L2). Path=["Intro"].
    
    assert len(segments) >= 2
    intro = segments[0] 
    details = segments[1]
    
    assert "Intro" in intro.content
    assert "Draft Header" in intro.content 
    assert details.level == 2
    assert details.path == ["Intro"]

# F-06: Section Pattern Override
def test_section_override(mock_loader_factory):
    # Text "Special Section" has No Style, but matches pattern Level 1.
    segments = mock_loader_factory(
        [
            (None, "Start"),
            (None, "Special Section"),
            (None, "Body Content")
        ],
        section_level_patterns=[(1, "Special Section")]
    )
    
    # 0. Root ("Start") - Preserved because not empty
    # 1. Special Section (L1)
    
    assert len(segments) == 2
    special = segments[1]
    assert special.level == 1
    assert "Special Section" in special.metadata["title"]
    assert "Body Content" in special.content

# F-07: Structured Leaf Concatenation
def test_structured_leaf_concat(mock_loader_factory):
    # H1 Parent -> H2 Child
    segments = mock_loader_factory(
        [
            ("Heading1", "Parent"),
            (None, "Parent Text"),
            ("Heading2", "Child"),
            (None, "Child Text")
        ],
        include_parent_content=True
    )
    
    # Root filtered (empty)
    child = segments[1] # 0=Parent, 1=Child
    assert "Parent" in child.content
    assert "Parent Text" in child.content
    assert "Child" in child.content
    assert "Child Text" in child.content
