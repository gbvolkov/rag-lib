from unittest.mock import MagicMock, patch

from rag_lib.chunkers.language import detect_nltk_language
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.chunkers.sentence import SentenceSplitter


def test_detect_nltk_language_prefers_cyrillic_text():
    text = "Привет, как дела? Это русский текст."
    assert detect_nltk_language(text) == "russian"


def test_sentence_splitter_auto_passes_detected_language_to_nltk():
    splitter = SentenceSplitter(language="auto")
    with patch("rag_lib.chunkers.sentence.sent_tokenize", return_value=["S1.", "S2."]) as mock_sent:
        splitter.split_text("Привет. Мир.")
    assert mock_sent.call_args.kwargs.get("language") == "russian"


def test_semantic_chunker_auto_passes_detected_language_to_nltk():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]

    chunker = SemanticChunker(
        embeddings=mock_embeddings,
        threshold=0.5,
        language="auto",
    )

    with patch("rag_lib.chunkers.semantic.sent_tokenize", return_value=["S1.", "S2."]) as mock_sent:
        chunker.split_text("Привет. Мир.")
    assert mock_sent.call_args.kwargs.get("language") == "russian"
