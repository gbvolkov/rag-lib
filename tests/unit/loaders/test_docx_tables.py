import pytest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.core.domain import SegmentType

# Sample XML for a 2x2 Table
# Row 1: Header 1, Header 2
# Row 2: Value 1, Value 2
TABLE_XML = """
<w:tbl xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:tr>
        <w:tc><w:p><w:r><w:t>Header 1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Header 2</w:t></w:r></w:p></w:tc>
    </w:tr>
    <w:tr>
        <w:tc><w:p><w:r><w:t>Value 1</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Value 2</w:t></w:r></w:p></w:tc>
    </w:tr>
</w:tbl>
"""

# Sample XML for a Paragraph
PARA_XML = """
<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:r><w:t>Normal text paragraph</w:t></w:r>
</w:p>
"""

@pytest.fixture
def mock_loader():
    """Returns a loader with mocked zip handling."""
    with patch("zipfile.ZipFile"):
        loader = StructuredLoader("dummy.docx")
        # We need to expose a way to test `_process_element` validly 
        # normally this is internal, but for unit testing logic we can call it 
        # if we mock the context properly.
        return loader

def test_extract_table_as_markdown(mock_loader):
    """Cycle 3.2: Verify a table element is converted to Markdown Segment."""
    # Setup namespaces normally done in __init__ or load
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    # Parse the XML fragment
    table_element = ET.fromstring(TABLE_XML)
    
    # We need to inject this element into the processing loop.
    # Since `load` does the zip reading, we might want to refactor `_process_element`
    # to be static or standalone, OR we mock the yield generator.
    
    # Let's assume we invoke a method `_parse_table(element)` -> str (markdown)
    # This method doesn't exist yet (Red phase).
    
    # Let's call the public method `_process_content` if we refactor `load` to use it,
    # or just test the new private method `_parse_table`.
    
    markdown = mock_loader._parse_table(table_element, ns)
    
    expected_md = "| Header 1 | Header 2 |\n|---|---|\n| Value 1 | Value 2 |"
    
    # Identify: We might need to handle whitespace/formatting exactly
    assert "| Header 1 | Header 2 |" in markdown
    assert "| Value 1 | Value 2 |" in markdown

def test_loader_integration_flow():
    """Verify load() identifies w:tbl and emits TABLE segment."""
    # This mocks the entire document.xml structure
    full_xml = f"""
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
            {PARA_XML}
            {TABLE_XML}
        </w:body>
    </w:document>
    """
    
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip.return_value.__enter__.return_value.read.return_value = full_xml.encode('utf-8')
        
        loader = StructuredLoader("dummy.docx")
        segments = loader.load()
        
        # Check we got 2 segments
        assert len(segments) == 2
        
        # Seg 1: Text
        assert segments[0].type == SegmentType.TEXT
        assert "Normal text paragraph" in segments[0].content
        
        # Seg 2: Table
        assert segments[1].type == SegmentType.TABLE
        assert "| Header 1 | Header 2 |" in segments[1].content
        assert segments[1].original_format == "markdown"
