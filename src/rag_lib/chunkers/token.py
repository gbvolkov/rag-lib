from typing import List, Callable, Optional
import tiktoken
from rag_lib.chunkers.base import TextSplitter

class TokenTextSplitter(TextSplitter):
    """
    Splits text into chunks that fit within a specified token limit.
    Uses tiktoken for encoding/decoding.
    """
    def __init__(
        self,
        chunk_size: int = 4000, 
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len, # Ignored in favor of token counting, but kept for signature compatibility
        model_name: str = "cl100k_base", 
        encoding_name: Optional[str] = None
    ):
        super().__init__(chunk_size, chunk_overlap, length_function)
        self.model_name = model_name
        self._encoding_name = encoding_name or model_name
        
        try:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except ValueError:
            # Fallback: try looking up by model name if encoding name failed or wasn't provided directly as an encoding
            try:
                self._encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                 raise ValueError(f"Could not find encoding for model {model_name} or encoding {encoding_name}")

    def split_text(self, text: str) -> List[str]:
        # usage of self._length_function is bypassed here in favor of token counting
        
        tokens = self._encoding.encode(text)
        
        if not tokens:
            return []
            
        chunks = []
        start_idx = 0
        total_tokens = len(tokens)
        
        while start_idx < total_tokens:
            end_idx = min(start_idx + self._chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end_idx == total_tokens:
                break
                
            # Move start_idx forward, subtracting overlap
            # Ensure we make progress
            step = self._chunk_size - self._chunk_overlap
            if step <= 0:
                 raise ValueError(f"Chunk overlap ({self._chunk_overlap}) must be smaller than chunk size ({self._chunk_size})")
            
            start_idx += step
            
        return chunks
