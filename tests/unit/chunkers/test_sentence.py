import pytest
from unittest.mock import patch

from rag_lib.chunkers.sentence import SentenceSplitter

def test_sentence_split_basic():
    """T-05: Basic sentence splitting without overflow."""
    text = "Sentence one. Sentence two. Sentence three."
    
    # Chunk size large enough for 2 sentences
    splitter = SentenceSplitter(chunk_size=30, chunk_overlap=0)
    chunks = splitter.split_text(text)
    
    # Expected: ["Sentence one. Sentence two.", "Sentence three."]
    # Lengths: ~27 chars, ~15 chars
    assert len(chunks) == 2
    assert "Sentence one. Sentence two." in chunks
    assert "Sentence three." in chunks

def test_sentence_split_overflow():
    """T-05: Handle sentence longer than chunk_size."""
    long_sentence = "A" * 50
    splitter = SentenceSplitter(chunk_size=20, chunk_overlap=0)
    chunks = splitter.split_text(long_sentence)
    
    # Should be split into 3 chunks (20, 20, 10)
    assert len(chunks) == 3
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 20
    assert len(chunks[2]) == 10

def test_sentence_split_overlap():
    """T-05: Basic overlap logic."""
    text = "S1. S2. S3. S4."
    # Each is ~3 chars. Total ~15 chars.
    # Chunk size 7 -> 2 sentences fit (3+1+3 = 7)
    # Overlap 3 -> 1 sentence overlap?
    
    splitter = SentenceSplitter(chunk_size=8, chunk_overlap=4)
    chunks = splitter.split_text(text)
    
    # "S1. S2." (7 chars)
    # Overlap "S2." (3 chars) -> Start next with "S2. S3."?
    # Logic in port: 
    # Current: [S1, S2]
    # Overlap: [S2] (len 3 <= 4)
    # Next: [S2, S3] (len 7)
    # Next: [S3, S4] (len 7)
    
    assert "S1. S2." in chunks
    assert "S2. S3." in chunks
    assert "S3. S4." in chunks


def test_sentence_split_uses_local_punkt_when_nltk_resource_is_missing():
    text = "Sentence one. Sentence two."
    splitter = SentenceSplitter(chunk_size=200, chunk_overlap=0)

    class _FakePunkt:
        def __init__(self, train_text: str):
            self.train_text = train_text

        def tokenize(self, source_text: str):
            assert self.train_text == text
            assert source_text == text
            return ["Sentence one.", "Sentence two."]

    with patch("rag_lib.chunkers.sentence.sent_tokenize", side_effect=LookupError("missing")), patch(
        "rag_lib.chunkers.sentence.PunktSentenceTokenizer",
        _FakePunkt,
    ):
        chunks = splitter.split_text(text)

    assert chunks == ["Sentence one. Sentence two."]
