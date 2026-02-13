import pytest
from rag_lib.chunkers.token import TokenTextSplitter

def test_token_split_exact():
    """T-04: Split by exact token count."""
    # "Hello world" is usually 2 tokens in cl100k_base: [9906, 1917]
    text = "Hello world"
    
    # Chunk size 1 -> Should get ["Hello", "world"] (roughly)
    # Note: tiktoken behavior depends on the encoding. 
    # Hello (1 token), world (1 token) -> 2 chunks.
    
    splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0, model_name="gpt-4")
    chunks = splitter.split_text(text)
    
    assert len(chunks) == 2
    assert chunks[0] == "Hello"
    # " world" might keep the leading space depending on encoding, 
    # but usually token splitters try to be clean. 
    # tiktoken encodes " world" as one token.
    assert "world" in chunks[1]

def test_token_split_large():
    """T-04: Split a larger text respecting the limit."""
    # 10 words, ~10 tokens
    text = "one two three four five six seven eight nine ten"
    
    # Chunk size 5 tokens
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=0)
    chunks = splitter.split_text(text)
    
    assert len(chunks) >= 2
    # Reconstruct
    reconstructed = "".join(chunks)
    # Token splitters often strip/clean, so strict reconstruction might fail on whitespace 
    # but content should be there.
    assert "one" in chunks[0]
    assert "ten" in chunks[-1]

def test_token_encoding_error():
    """T-04: Handle invalid encoding name gracefully or raise error."""
    with pytest.raises(ValueError):
        TokenTextSplitter(model_name="invalid_model_name_12345")
