import pytest
from unittest.mock import MagicMock, patch, mock_open
import xml.etree.ElementTree as ET
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.chunkers.markdown_table import MarkdownTableSplitter
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.core.domain import Segment, SegmentType

# --- Semantic Chunker Tests ---

def test_semantic_chunker_basic_logic():
    mock_embeddings = MagicMock()
    # 3 sentences. 
    # Sim(0,1) = 0.9 (keep together)
    # Sim(1,2) = 0.1 (split)
    # Result: [Sent0+Sent1, Sent2]
    
    # embed_documents returns list of vectors.
    # We need 3 vectors.
    # v0=[1,0], v1=[0.9, 0.1], v2=[0,1]
    # Cosine(v0,v1) ~ high. Cosine(v1,v2) ~ low.
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0],
        [0.9, 0.1], 
        [0.0, 1.0] 
    ]
    
    chunker = SemanticChunker(embeddings=mock_embeddings, threshold=0.5)
    
    text = "Sentence A. Sentence B. Sentence C."
    # We need to patch sent_tokenize because it might split differently or need download
    with patch("rag_lib.chunkers.semantic.sent_tokenize") as mock_sent:
        mock_sent.return_value = ["Sentence A.", "Sentence B.", "Sentence C."]
        
        segments = chunker.split_text(text)
        
    assert len(segments) == 2
    assert "Sentence A. Sentence B." in segments[0].content
    assert "Sentence C." in segments[1].content

def test_semantic_chunker_empty_input():
    mock_embeddings = MagicMock()
    chunker = SemanticChunker(embeddings=mock_embeddings)
    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=[]):
        segments = chunker.split_text("")
    assert len(segments) == 0

# --- Markdown Table Splitter Tests ---

def test_markdown_splitter_mixed_content():
    text = """
    Preamble text.
    
    | H1 | H2 |
    |---|---|
    | V1 | V2 |
    
    Postamble text.
    """
    splitter = MarkdownTableSplitter()
    segments = splitter.split_text(text)
    
    # Should get 3 segments: Text, Table, Text
    # Regex might correspond to 3 matches? No, 3 segments.
    # finditer finds table. Splitter adds "before" text, then table.
    
    assert len(segments) == 3
    assert segments[0].type == SegmentType.TEXT
    assert "Preamble" in segments[0].content
    
    assert segments[1].type == SegmentType.TABLE
    assert "| H1 | H2 |" in segments[1].content
    
    assert segments[2].type == SegmentType.TEXT
    assert "Postamble" in segments[2].content

def test_markdown_splitter_no_table():
    text = "Just some text."
    splitter = MarkdownTableSplitter()
    segments = splitter.split_text(text)
    
    assert len(segments) == 1
    assert segments[0].type == SegmentType.TEXT
    assert segments[0].content == "Just some text."

# --- Structured Loader (DOCX) Tests ---

def test_structured_loader_docx_parsing():
    # Mock zipfile to return XML
    xml_content = b"""
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
            <w:p>
                <w:pPr>
                    <w:pStyle w:val="Heading1"/>
                </w:pPr>
                <w:r><w:t>Chapter 1</w:t></w:r>
            </w:p>
            <w:p>
                <w:r><w:t>Body text.</w:t></w:r>
            </w:p>
        </w:body>
    </w:document>
    """
    
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip_instance = MagicMock()
        mock_zip_instance.read.return_value = xml_content
        mock_zip.return_value.__enter__.return_value = mock_zip_instance
        
        loader = StructuredLoader("dummy.docx")
        segments = loader.load()
        
    assert len(segments) == 1 # Heading 1 starts a segment that likely contains Body text.
    # Logic: Heading 1 found -> New segment. Body text -> appended to current segment.
    
    seg = segments[0]
    assert "Chapter 1" in seg.metadata["title"]
    assert "Body text." in seg.content
    assert seg.level == 1

def test_structured_loader_bad_zip():
    import zipfile
    with patch("zipfile.ZipFile") as mock_zip:
        mock_zip.side_effect = zipfile.BadZipFile("Not a valid zip")
        
        loader = StructuredLoader("bad.docx")
        with pytest.raises(zipfile.BadZipFile):
            loader.load()
