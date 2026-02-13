import numpy as np
from typing import List, Dict
from langchain_core.embeddings import Embeddings

class MockEmbeddings(Embeddings):
    """
    A deterministic mock embedding model for testing.
    Assigns pre-defined vectors to known words/concepts to simulate semantic similarity.
    Everything else gets a random (stable) vector.
    """
    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        # Define some "concepts" with orthogonal vectors for testing
        # 4D vectors for easy mental math: [Concept1, Concept2, Concept3, Noise]
        self.concepts: Dict[str, np.ndarray] = {
            "apple": np.array([1.0, 0.1, 0.0, 0.0]),
            "banana": np.array([0.9, 0.2, 0.0, 0.0]), # Close to apple
            "dog": np.array([0.0, 1.0, 0.1, 0.0]),
            "cat": np.array([0.0, 0.9, 0.2, 0.0]),    # Close to dog, far from apple
            "space": np.array([0.0, 0.0, 1.0, 0.1]),
            "planet": np.array([0.0, 0.0, 0.9, 0.2]), # Close to space
        }

    def _get_vector(self, text: str) -> List[float]:
        text_lower = text.lower()
        # Check if any concept word is in the text
        for concept, vector in self.concepts.items():
            if concept in text_lower:
                # Add some slight noise based on string hash to make it unique but stable
                seed = sum(ord(c) for c in text) 
                noise = np.sin(seed) * 0.01
                return (vector + noise).tolist()
        
        # Fallback: pseudo-random based on text hash
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(self.dimension).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_vector(text)
