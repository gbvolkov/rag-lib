import numpy as np
from typing import Any, Dict, List, Optional
from rag_lib.core.domain import Segment, SegmentType
from langchain_core.embeddings import Embeddings 
import nltk
from nltk.tokenize import sent_tokenize

from rag_lib.config import Settings
from rag_lib.core.logger import logger
from rag_lib.chunkers.base import TextSplitter

class SemanticChunker(TextSplitter):
    """
    Splits text based on semantic similarity of sentences.
    Inherits from TextSplitter to support standardized pipeline.
    """
    def __init__(
        self,
        embeddings: Embeddings,
        threshold: Optional[float] = None,
        window_size: int = 1,
        threshold_type: str = "fixed",
        percentile_threshold: int = 90,
        local_percentile_window: int = 50,
        local_min_samples: int = 20,
        local_fallback: str = "global",
        enable_debug: bool = False,
    ):
        """
        Args:
            embeddings: LangChain Embeddings model.
            threshold: Fixed cosine similarity threshold. Defaults to config if None.
            window_size: Number of sentences to look ahead when building contextual vectors.
                        window_size=1 compares adjacent sentence embeddings directly.
            threshold_type: "fixed", "percentile", or "percentile_local".
            percentile_threshold: If "percentile", the percentile of similarity to use as threshold.
            local_percentile_window: For "percentile_local", number of boundaries to include
                                     on each side of current boundary when computing local percentile.
            local_min_samples: Minimum local sample size required, otherwise fallback is used.
            local_fallback: Fallback threshold source for local mode when local sample is too small.
                            Allowed: "global", "fixed".
            enable_debug: If True, stores boundary-level diagnostics for the last split_text() call.
        """
        super().__init__() # Initialize base (chunk_size/overlap not typically used here but good practice)
        
        if threshold is None:
            threshold = Settings().ingestion.semantic_threshold
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if threshold_type not in {"fixed", "percentile", "percentile_local"}:
            raise ValueError("threshold_type must be one of: fixed, percentile, percentile_local")
        if local_percentile_window < 1:
            raise ValueError("local_percentile_window must be >= 1")
        if local_min_samples < 1:
            raise ValueError("local_min_samples must be >= 1")
        if local_fallback not in {"global", "fixed"}:
            raise ValueError("local_fallback must be one of: global, fixed")

        self.embeddings = embeddings
        self.threshold = threshold
        self.window_size = window_size
        self.threshold_type = threshold_type
        self.percentile_threshold = percentile_threshold
        self.local_percentile_window = local_percentile_window
        self.local_min_samples = local_min_samples
        self.local_fallback = local_fallback
        self.enable_debug = enable_debug
        self._last_debug_info: Optional[Dict[str, Any]] = None

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _build_lookahead_vectors(self, sentence_embeddings: List[List[float]]) -> List[np.ndarray]:
        """
        Builds contextual vectors with look-ahead windows.
        For sentence i, we average embeddings[i : i + window_size].
        """
        vectors: List[np.ndarray] = []
        total = len(sentence_embeddings)
        for i in range(total):
            end = min(i + self.window_size, total)
            window = np.array(sentence_embeddings[i:end])
            vectors.append(np.mean(window, axis=0))
        return vectors

    def _compute_local_threshold(
        self,
        similarities: List[float],
        boundary_index: int,
        global_threshold: float,
    ) -> tuple[float, str, int, int]:
        """
        Compute local percentile threshold around boundary_index.
        Returns: (threshold, source, local_start, local_end_exclusive)
        """
        local_start = max(0, boundary_index - self.local_percentile_window)
        local_end = min(len(similarities), boundary_index + self.local_percentile_window + 1)
        local_sample = similarities[local_start:local_end]

        if len(local_sample) < self.local_min_samples:
            if self.local_fallback == "fixed":
                return float(self.threshold), "fixed_fallback", local_start, local_end
            return global_threshold, "global_fallback", local_start, local_end

        local_threshold = float(np.percentile(local_sample, self.percentile_threshold))
        return local_threshold, "local_percentile", local_start, local_end

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into semantically grouped sentence chunks (strings).
        """
        self._last_debug_info = None
        self._ensure_sentence_tokenizer()

        # 1. Split into Sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            if self.enable_debug:
                self._last_debug_info = {
                    "window_size": self.window_size,
                    "threshold_type": self.threshold_type,
                    "configured_threshold": float(self.threshold),
                    "percentile_threshold": self.percentile_threshold if self.threshold_type in {"percentile", "percentile_local"} else None,
                    "local_percentile_window": self.local_percentile_window if self.threshold_type == "percentile_local" else None,
                    "local_min_samples": self.local_min_samples if self.threshold_type == "percentile_local" else None,
                    "local_fallback": self.local_fallback if self.threshold_type == "percentile_local" else None,
                    "effective_threshold": float(self.threshold),
                    "sentence_count": len(sentences),
                    "chunk_count": len(sentences),
                    "split_count": 0,
                    "similarities": [],
                    "boundaries": [],
                    "split_boundaries": [],
                }
            return sentences

        # 2. Calculate Similarities
        # For efficiency, we embed all sentences at once
        sentence_embeddings = self.embeddings.embed_documents(sentences)
        contextual_vectors = self._build_lookahead_vectors(sentence_embeddings)

        similarities: List[float] = []
        for i in range(len(contextual_vectors) - 1):
            # Cosine similarity between adjacent contextual vectors.
            vec1 = contextual_vectors[i]
            vec2 = contextual_vectors[i + 1]
            sim = float(self._cosine_similarity(vec1, vec2))
            similarities.append(sim)

        # 2.5 Determine global threshold (used directly for fixed/global-percentile
        # and as fallback/reference for local percentile mode).
        global_threshold = float(self.threshold)
        if self.threshold_type in {"percentile", "percentile_local"} and similarities:
            global_threshold = float(np.percentile(similarities, self.percentile_threshold))
        
        # 5. Group
        chunks: List[str] = []
        current_group = [sentences[0]]
        boundaries: List[Dict[str, Any]] = []
        split_boundaries: List[Dict[str, Any]] = []
        
        for i in range(1, len(sentences)):
            # sim index i-1 corresponds to pair (sentences[i-1], sentences[i])
            boundary_index = i - 1
            sim = similarities[i - 1]
            current_chunk_index = len(chunks)

            threshold_source = "fixed"
            local_range_start: Optional[int] = None
            local_range_end: Optional[int] = None

            if self.threshold_type == "fixed":
                boundary_threshold = float(self.threshold)
            elif self.threshold_type == "percentile":
                boundary_threshold = global_threshold
                threshold_source = "global_percentile"
            else:
                boundary_threshold, threshold_source, local_range_start, local_range_end = self._compute_local_threshold(
                    similarities=similarities,
                    boundary_index=boundary_index,
                    global_threshold=global_threshold,
                )

            is_join = sim >= boundary_threshold

            left_window_start = i - 1
            left_window_end = min(left_window_start + self.window_size, len(sentences))
            right_window_start = i
            right_window_end = min(right_window_start + self.window_size, len(sentences))

            boundary_debug: Dict[str, Any] = {
                "boundary_index": boundary_index,
                "left_sentence_index": i - 1,
                "right_sentence_index": i,
                "left_sentence": sentences[i - 1],
                "right_sentence": sentences[i],
                "left_window_sentence_indices": list(range(left_window_start, left_window_end)),
                "right_window_sentence_indices": list(range(right_window_start, right_window_end)),
                "left_window_text": " ".join(sentences[left_window_start:left_window_end]),
                "right_window_text": " ".join(sentences[right_window_start:right_window_end]),
                "similarity": sim,
                "threshold": boundary_threshold,
                "threshold_source": threshold_source,
                "local_similarity_range": [local_range_start, local_range_end] if local_range_start is not None else None,
                "global_threshold_reference": global_threshold,
                "decision": "join" if is_join else "split",
            }
            
            if is_join:
                # Same topic
                current_group.append(sentences[i])
                boundary_debug["left_chunk_index"] = current_chunk_index
                boundary_debug["right_chunk_index"] = current_chunk_index
            else:
                # Split
                chunks.append(" ".join(current_group))
                current_group = [sentences[i]]
                boundary_debug["left_chunk_index"] = current_chunk_index
                boundary_debug["right_chunk_index"] = current_chunk_index + 1
                split_boundaries.append(boundary_debug)

            boundaries.append(boundary_debug)
        
        # Flush last group
        if current_group:
            chunks.append(" ".join(current_group))

        if self.enable_debug:
            self._last_debug_info = {
                "window_size": self.window_size,
                "threshold_type": self.threshold_type,
                "configured_threshold": float(self.threshold),
                "percentile_threshold": self.percentile_threshold if self.threshold_type in {"percentile", "percentile_local"} else None,
                "local_percentile_window": self.local_percentile_window if self.threshold_type == "percentile_local" else None,
                "local_min_samples": self.local_min_samples if self.threshold_type == "percentile_local" else None,
                "local_fallback": self.local_fallback if self.threshold_type == "percentile_local" else None,
                "effective_threshold": global_threshold if self.threshold_type != "percentile_local" else None,
                "global_effective_threshold": global_threshold,
                "sentence_count": len(sentences),
                "chunk_count": len(chunks),
                "split_count": len(split_boundaries),
                "similarities": similarities,
                "boundaries": boundaries,
                "split_boundaries": split_boundaries,
            }
        
        return chunks

    def get_last_debug_info(self) -> Optional[Dict[str, Any]]:
        """
        Returns boundary-level diagnostics collected during the last split_text call.
        """
        return self._last_debug_info

    def get_split_boundary_for_chunk(self, left_chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        Returns split diagnostics for boundary between left_chunk_index and left_chunk_index + 1.
        """
        if not self._last_debug_info:
            return None
        for boundary in self._last_debug_info.get("split_boundaries", []):
            if boundary.get("left_chunk_index") == left_chunk_index:
                return boundary
        return None

    def _ensure_sentence_tokenizer(self) -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError as exc:
            msg = (
                "NLTK tokenizer 'punkt' is required for SemanticChunker. "
                "Install it manually with: python -m nltk.downloader punkt"
            )
            raise LookupError(msg) from exc
