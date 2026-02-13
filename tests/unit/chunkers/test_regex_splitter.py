import pytest
from rag_lib.chunkers.regex import RegexSplitter

def test_regex_split_numbering():
    """T-03: Split by numbering pattern (keeping delimiter)."""
    text = "1. Introduction text. 2. Body text. 3. Conclusion."
    
    # Lookahead pattern to split *before* the number, keeping it in the next chunk
    pattern = r"(?=\d+\.\s)"
    
    splitter = RegexSplitter(pattern=pattern)
    chunks = splitter.split_text(text)
    
    # Expected behavior: 
    # re.split gives an empty first element if the string starts with the match
    # The splitter should ideally filter empty chunks.
    
    assert len(chunks) == 3
    assert chunks[0] == "1. Introduction text. "
    assert chunks[1] == "2. Body text. "
    assert chunks[2] == "3. Conclusion."

def test_regex_split_qa():
    """T-03: Split by Q/A patterns."""
    text = "Q: What is X?\nA: X is Y.\nQ: Why?\nA: Because."
    
    # Split before "Q:"
    pattern = r"(?=Q:)"
    
    splitter = RegexSplitter(pattern=pattern)
    chunks = splitter.split_text(text)
    
    # Check cleaning/stripping if applicable, or exact content
    assert len(chunks) == 2
    assert "Q: What is X?" in chunks[0]
    assert "Q: Why?" in chunks[1]

def test_regex_no_match():
    """T-03: Fallback when no pattern matches."""
    text = "Just some text without numbers."
    pattern = r"(?=\d+\.\s)"
    
    splitter = RegexSplitter(pattern=pattern)
    chunks = splitter.split_text(text)
    
    assert len(chunks) == 1
    assert chunks[0] == text

def test_complex_multipattern_split():
    """
    Complex Scenario: Split on multiple delimiters (Sections, Notes, Warnings)
    using a combined regex pattern.
    Mimics a document segment containing heterogeneous sub-blocks.
    """
    text = (
        "General introduction text.\n"
        "Section 1. Core Logic.\n"
        "Some details here.\n"
        "Note: specific edge case.\n"
        "Section 2. Advanced Logic.\n"
        "More details.\n"
        "Warning: This is dangerous."
    )
    
    # Split on "Section <num>." OR "Note:" OR "Warning:"
    # Uses non-capturing group (?:...) inside lookahead (?=...) 
    # to maintain the delimiter at the start of the chunk.
    pattern = r"(?=(?:Section \d+\.|Note:|Warning:))"
    
    splitter = RegexSplitter(pattern=pattern)
    chunks = splitter.split_text(text)
    
    # Expected chunks:
    # 0: "General introduction text.\n" (Pre-header content)
    # 1: "Section 1..."
    # 2: "Note:..."
    # 3: "Section 2..."
    # 4: "Warning:..."
    
    assert len(chunks) == 5
    assert chunks[0].strip() == "General introduction text."
    assert chunks[1].startswith("Section 1")
    assert "Core Logic" in chunks[1]
    assert chunks[2].startswith("Note:")
    assert chunks[3].startswith("Section 2")
    assert chunks[4].startswith("Warning:")
