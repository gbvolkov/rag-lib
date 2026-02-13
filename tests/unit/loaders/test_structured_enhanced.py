import pytest
from unittest.mock import patch
from rag_lib.loaders.structured import StructuredLoader

# Helper to create mock docx XML with complex structure
def create_complex_xml():
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    
    # H1: Contract
    xml += '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Service Agreement</w:t></w:r></w:p>'
    
    # H2: Definitions
    xml += '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>1. Definitions</w:t></w:r></w:p>'
    xml += '<w:p><w:r><w:t>"Service" means...</w:t></w:r></w:p>'
    
    # H2: Terms (with Regex "Clause" inside)
    xml += '<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>2. Terms</w:t></w:r></w:p>'
    xml += '<w:p><w:r><w:t>General terms apply.</w:t></w:r></w:p>'
    xml += '<w:p><w:r><w:t>Clause 2.1: Payment is due net 30.</w:t></w:r></w:p>'
    xml += '<w:p><w:r><w:t>Clause 2.2: Late fees apply.</w:t></w:r></w:p>'
    
    xml += '</w:body></w:document>'
    return xml.encode('utf-8')

@pytest.fixture
def mock_docx_complex():
    with patch('zipfile.ZipFile') as mock_zip:
        instance = mock_zip.return_value.__enter__.return_value
        instance.read.return_value = create_complex_xml()
        yield instance

def test_complex_structured_mix(mock_docx_complex):
    # Regex pattern to catch "Clause X.Y" inside H2 segments
    regex_patterns = [(3, r"Clause \d+\.\d+")] 
    
    loader = StructuredLoader("contract.docx", regex_patterns=regex_patterns)
    segments = loader.load()
    
    # Expected Structure:
    # 1. H1 Service Agreement
    # 2. H2 Definitions
    # 3. H2 Terms -> Contains sub-segments for clauses
    
    # Identify Clause 2.1
    clause_2_1 = next(s for s in segments if "Clause 2.1" in s.content)
    
    # Path should include H1 and H2
    # H1 "Service Agreement" -> H2 "2. Terms" -> Clause
    # Note: Flattened list of titles
    assert "Service Agreement" in clause_2_1.path
    assert "2. Terms" in clause_2_1.path
    assert clause_2_1.level == 3 # From regex pattern
