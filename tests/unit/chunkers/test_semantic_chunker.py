import pytest
from unittest.mock import MagicMock, patch
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.embeddings.mock import MockEmbeddings
from rag_lib.core.domain import SegmentType

def test_semantic_chunking_topic_shift():
    # Setup Mock Embeddings with knwon concepts
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.6) # Threshold is 1 - distance, or similarity directly? Usually similarity.
    # If using similarity: High means same topic. Break if < 0.6.
    
    # Text with two distinct topics: Fruit and Pets
    # "apple" and "banana" -> High Similarity
    # "dog" and "cat" -> High Similarity
    # "banana" vs "dog" -> Low Similarity
    text = (
        "I like to eat apple. "
        "Apple is a tasty fruit. "
        "Banana is yellow and sweet. "  # End of Topic 1
        "I have a pet dog. "            # Start of Topic 2 (Break expected here)
        "The dog barks at the cat. "
        "Cat loves to sleep."
    )
    
    segments = chunker.chunk(text)
    
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    
    # Check contents
    assert "apple" in segments[0].content.lower()
    assert "banana" in segments[0].content.lower()
    assert "dog" not in segments[0].content.lower()
    
    assert "dog" in segments[1].content.lower()
    assert "cat" in segments[1].content.lower()
    assert "apple" not in segments[1].content.lower()
    
    assert segments[0].type == SegmentType.TEXT

def test_semantic_chunking_single_topic():
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(embeddings=embeddings, threshold=0.5)
    
    text = "Apple is red. I like apple. Apple pie is good."
    segments = chunker.chunk(text)
    
    assert len(segments) == 1

def test_semantic_chunking_window_size_lookahead_changes_behavior():
    # S1 and S3 are similar, S2 differs.
    # With window_size=1: adjacent sims are low -> splits.
    # With window_size=2: look-ahead contextual vectors become similar -> one chunk.
    sentence_vectors = [
        [1.0, 0.0],  # S1
        [0.0, 1.0],  # S2
        [1.0, 0.0],  # S3
    ]
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = sentence_vectors
    text = "S1. S2. S3."

    chunker_w1 = SemanticChunker(embeddings=mock_embeddings, threshold=0.6, window_size=1)
    chunker_w2 = SemanticChunker(embeddings=mock_embeddings, threshold=0.6, window_size=2)

    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=["S1.", "S2.", "S3."]):
        chunks_w1 = chunker_w1.split_text(text)
        chunks_w2 = chunker_w2.split_text(text)

    assert len(chunks_w1) == 3
    assert len(chunks_w2) == 1

def test_semantic_chunker_debug_boundary_trace():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    chunker = SemanticChunker(
        embeddings=mock_embeddings,
        threshold=0.6,
        window_size=1,
        enable_debug=True,
    )

    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=["S1.", "S2.", "S3."]):
        chunks = chunker.split_text("S1. S2. S3.")

    assert len(chunks) == 3

    debug_info = chunker.get_last_debug_info()
    assert debug_info is not None
    assert debug_info["split_count"] == 2
    assert debug_info["effective_threshold"] == 0.6

    boundary = chunker.get_split_boundary_for_chunk(0)
    assert boundary is not None
    assert boundary["decision"] == "split"
    assert boundary["left_chunk_index"] == 0
    assert boundary["right_chunk_index"] == 1
    assert boundary["left_sentence"] == "S1."
    assert boundary["right_sentence"] == "S2."


class _PredefinedSimilarityChunker(SemanticChunker):
    def __init__(self, similarities, sentence_count, **kwargs):
        self._similarities = list(similarities)
        self._sim_idx = 0
        fake_embeddings = MagicMock()
        fake_embeddings.embed_documents.return_value = [[0.0, 0.0] for _ in range(sentence_count)]
        super().__init__(embeddings=fake_embeddings, **kwargs)

    def _cosine_similarity(self, v1, v2) -> float:
        val = self._similarities[self._sim_idx]
        self._sim_idx += 1
        return val


def test_semantic_chunking_percentile_local_uses_local_context():
    # 10 sentences => 9 boundary similarities.
    # Global percentile is dominated by high values, but local window around boundary 4
    # contains lower values and should allow joining there.
    sims = [0.98, 0.98, 0.98, 0.40, 0.42, 0.41, 0.98, 0.98, 0.98]
    sentences = [f"S{i}." for i in range(1, 11)]

    global_chunker = _PredefinedSimilarityChunker(
        similarities=sims,
        sentence_count=len(sentences),
        threshold_type="percentile",
        percentile_threshold=60,
    )
    local_chunker = _PredefinedSimilarityChunker(
        similarities=sims,
        sentence_count=len(sentences),
        threshold_type="percentile_local",
        percentile_threshold=60,
        local_percentile_window=1,
        local_min_samples=3,
        enable_debug=True,
    )

    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=sentences):
        global_chunks = global_chunker.split_text(" ".join(sentences))
        local_chunks = local_chunker.split_text(" ".join(sentences))

    assert len(global_chunks) == 4
    assert len(local_chunks) == 3

    debug_info = local_chunker.get_last_debug_info()
    assert debug_info is not None
    boundary_4 = next(b for b in debug_info["boundaries"] if b["boundary_index"] == 4)
    assert boundary_4["decision"] == "join"
    assert boundary_4["threshold_source"] == "local_percentile"


def test_semantic_chunking_percentile_local_fallback_fixed():
    sims = [0.6, 0.4]
    sentences = ["S1.", "S2.", "S3."]

    chunker = _PredefinedSimilarityChunker(
        similarities=sims,
        sentence_count=len(sentences),
        threshold_type="percentile_local",
        percentile_threshold=60,
        local_percentile_window=1,
        local_min_samples=10,  # Force fallback on short sequence
        local_fallback="fixed",
        threshold=0.5,
        enable_debug=True,
    )

    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=sentences):
        _ = chunker.split_text(" ".join(sentences))

    debug_info = chunker.get_last_debug_info()
    assert debug_info is not None
    first_boundary = debug_info["boundaries"][0]
    assert first_boundary["threshold_source"] == "fixed_fallback"
    assert first_boundary["threshold"] == 0.5

