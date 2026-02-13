from typing import List, Callable
import re
from rag_lib.chunkers.base import TextSplitter

class RegexSplitter(TextSplitter):
    """
    Splits text based on a provided regex pattern.
    Commonly used with lookahead/lookbehind to keep delimiters.
    """
    def __init__(
        self,
        pattern: str,
        chunk_size: int = 4000, 
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        super().__init__(chunk_size, chunk_overlap, length_function)
        self.pattern = pattern

    def split_text(self, text: str) -> List[str]:
        # Split using the regex
        splits = re.split(self.pattern, text)
        
        # Filter out empty strings that result from start/end matches or capture groups behavior
        valid_splits = [s for s in splits if s]
        
        return valid_splits
