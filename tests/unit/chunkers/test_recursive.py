import pytest
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter as LibRecursive
# Import LangChain for parity check
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LcRecursive
except ImportError:
    # Fallback to langchain.text_splitter if installed (though script ruled it out, safe to keep)
    from langchain.text_splitter import RecursiveCharacterTextSplitter as LcRecursive

def test_recursive_basic_logic():
    """T-02: Verify basic recursive splitting behavior."""
    text = "Para1\n\nPara2"
    splitter = LibRecursive(chunk_size=10, chunk_overlap=0, separators=["\n\n"])
    chunks = splitter.split_text(text)
    assert chunks == ["Para1", "Para2"]

def test_recursive_fallback():
    """T-02: Verify fallback to smaller separators."""
    # "Long line" (9 chars) + "\n" + "Short" (5 chars)
    # Total = 15. Chunk size = 10.
    # Should split on \n.
    text = "Long line\nShort"
    splitter = LibRecursive(chunk_size=10, chunk_overlap=0, separators=["\n\n", "\n"])
    chunks = splitter.split_text(text)
    assert chunks == ["Long line", "Short"]

def test_recursive_overlap():
    """T-02: Verify overlap integrity."""
    text = "ABCD"
    # Chunk size 2, overlap 1. e.g. AB, BC, CD?
    # Actually recursive splitter works by separators. 
    # If no separators match, it might split by char if empty string is in separators.
    splitter = LibRecursive(chunk_size=2, chunk_overlap=1, separators=[""])
    chunks = splitter.split_text(text)
    assert chunks == ["AB", "BC", "CD"]

def test_parity_with_langchain():
    """
    User Request: Compare results against LangChain standard text splitter.
    """
    text = (
        "This is a long paragraph that needs to be split. "
        "It represents a common scenario in RAG applications.\n\n"
        "Here is a second paragraph. It contains some details about the implementation "
        "of the recursive character text splitter."
    )
    
    chunk_size = 50
    chunk_overlap = 10
    separators = ["\n\n", "\n", " ", ""]
    
    # 1. Our Implementation
    lib_splitter = LibRecursive(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    lib_chunks = lib_splitter.split_text(text)
    
    # 2. LangChain Implementation
    lc_splitter = LcRecursive(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        # Ensure length function is the same (len)
        length_function=len,
        add_start_index=False
    )
    lc_chunks = lc_splitter.split_text(text)
    
    # Debug print if failure
    if lib_chunks != lc_chunks:
        print(f"\nLib Chunks ({len(lib_chunks)}): {lib_chunks}")
        print(f"LC  Chunks ({len(lc_chunks)}): {lc_chunks}")

    assert lib_chunks == lc_chunks
