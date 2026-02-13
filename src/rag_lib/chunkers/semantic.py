import numpy as np
from typing import List, Optional
from rag_lib.core.domain import Segment, SegmentType
from langchain_core.embeddings import Embeddings 
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is downloaded (similar to SentenceSplitter)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from rag_lib.config import Settings
from rag_lib.core.logger import logger
from rag_lib.chunkers.base import TextSplitter # Added logger import

class SemanticChunker(TextSplitter): # Changed inheritance to TextSplitter
    """
    Splits text based on semantic similarity of sentences.
    """
    def __init__(self, embeddings: Embeddings, threshold: Optional[float] = None, window_size: int = 1,
                 threshold_type: str = "fixed", percentile_threshold: int = 90):
        """
        Args:
            embeddings: LangChain Embeddings model.
            threshold: Fixed cosine similarity threshold. Defaults to config if None.
            window_size: Number of sentences to look ahead (currently 1 for adjacent).
            threshold_type: "fixed" or "percentile".
            percentile_threshold: If "percentile", the percentile of similarity to use as threshold.
        """
        if threshold is None:
            threshold = Settings().ingestion.semantic_threshold

        self.embeddings = embeddings
        self.threshold = threshold
        self.window_size = window_size
        self.threshold_type = threshold_type
        self.percentile_threshold = percentile_threshold

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def split_text(self, text: str) -> List[str]: # Renamed method to split_text and changed return type
        # 1. Split into Sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return sentences

        # 2. Calculate Similarities
        # For efficiency, we embed all sentences at once
        embeddings = self.embeddings.embed_documents(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            # Cosine similarity
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            sim = self._cosine_similarity(vec1, vec2)
            similarities.append(sim)

        # 2.5 Determine Threshold Dynamically
        current_threshold = self.threshold
        if self.threshold_type == "percentile" and similarities:
            import numpy as np
            current_threshold = np.percentile(similarities, self.percentile_threshold)
        
        # 5. Group
        segments: List[Segment] = []
        current_group = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # sim index i-1 corresponds to pair (sentences[i-1], sentences[i])
            sim = similarities[i-1]
            
            if sim >= current_threshold:
                # Same topic
                current_group.append(sentences[i])
            else:
                # Split
                segments.append(Segment(content=" ".join(current_group), type=SegmentType.TEXT))
                current_group = [sentences[i]]
        
        # Flush last group
        if current_group:
            segments.append(Segment(content=" ".join(current_group), type=SegmentType.TEXT))
            
        return segments
