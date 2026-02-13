import pytest
from unittest.mock import MagicMock, patch
from rag_lib.loaders.structured import StructuredLoader

# Helper to create mock docx XML
def create_docx_xml(paragraphs):
    # paragraphs is list of (style, text)
    # style "Heading1", "Heading2", or None
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    for style, text in paragraphs:
        xml += '<w:p>'
        if style:
            xml += f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        xml += f'<w:r><w:t>{text}</w:t></w:r></w:p>'
    xml += '</w:body></w:document>'
    return xml.encode('utf-8')

@pytest.fixture
def mock_docx_loader():
    with patch('zipfile.ZipFile') as mock_zip:
        yield mock_zip

# S-01: Mix Strategy (Regex inside Structure)
def test_mix_strategy(mock_docx_loader):
    # Setup Mock XML: H1 followed by text containing "Rule 1", "Rule 2"
    xml_content = create_docx_xml([
        ("Heading1", "Chapter 1"),
        (None, "Intro text."),
        (None, "Rule 1: Do X."),
        (None, "Rule 2: Do Y.")
    ])
    
    # Configure mock
    mock_zip_instance = mock_docx_loader.return_value.__enter__.return_value
    mock_zip_instance.read.return_value = xml_content
    
    # Loader with Regex pattern for "Rule \d"
    loader = StructuredLoader("dummy.docx", regex_patterns=[(2, r"Rule \d+")])
    segments = loader.load()
    
    # Expect: 
    # Seg 1: "Chapter 1\nIntro text." (Level 1)
    # Seg 2: "Rule 1: Do X." (Level 2? Or child of Ch1?)
    # Seg 3: "Rule 2: Do Y."
    
    # Logic: Regex splits the content of the structural segment.
    # Implementation should handle this recursion.
    
    print(f"\nDebug Structure: {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"Seg {i}: L{seg.level} Content='{seg.content}' Path={seg.path}")

    assert len(segments) >= 3
    assert segments[0].content.strip() == "Chapter 1\nIntro text."
    assert segments[1].content.strip() == "Rule 1: Do X."
    assert segments[1].path == ["Chapter 1"] # Inherits parent path?

# S-02: XML Hierarchy
def test_xml_hierarchy(mock_docx_loader):
    xml_content = create_docx_xml([
        ("Heading1", "Title"),
        (None, "Body"),
        ("Heading2", "Subtitle"),
        (None, "SubBody")
    ])
     
    mock_zip_instance = mock_docx_loader.return_value.__enter__.return_value
    mock_zip_instance.read.return_value = xml_content
    
    loader = StructuredLoader("dummy.docx")
    segments = loader.load()
    
    assert len(segments) == 2
    assert segments[0].content.strip() == "Title\nBody"
    assert segments[0].level == 1
    
    assert segments[1].content.strip() == "Subtitle\nSubBody"
    assert segments[1].path == ["Title"]
    assert segments[1].level == 2
