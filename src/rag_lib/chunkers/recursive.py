from typing import List, Optional, Callable, Any, Iterable
import re
from rag_lib.chunkers.base import TextSplitter

class RecursiveCharacterTextSplitter(TextSplitter):
    """
    Implementation of LangChain's RecursiveCharacterTextSplitter logic.
    Splits text by a list of separators, trying them in order until the chunks
    are small enough.
    
    Default separators: ["\n\n", "\n", " ", ""]
    """
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        separators: Optional[List[str]] = None,
        keep_separator: bool = False,
        is_separator_regex: bool = False,
    ):
        super().__init__(chunk_size, chunk_overlap, length_function)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._keep_separator = keep_separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursive splitting logic.
        """
        final_chunks = []
        
        # Determine which separator to use
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            _separator = _s
            if self._is_separator_regex:
                # If regex, check if it matches
                pass # Simplified for now, sticking to string matching primarily unless regex requested
            
            if _s == "":
                separator = _s
                break
                
            if _s in text:
                separator = _s
                new_separators = separators[i + 1 :]
                break
        else:
            # No separator found in text, treat as one chunk if generic
            pass

        # Split
        if separator:
             splits = text.split(separator)
        else:
             splits = list(text) # Split by char if empty string separator

        # Now merge splits that are too small
        good_splits = []
        _separator = separator if self._keep_separator else "" # We don't keep sep by default? 
        # Actually LangChain behavior:
        # If splitting by "\n\n", we usually lose it.
        # But let's follow the standard "split" behavior of python strings which removes it.
        
        current_doc = []
        total = 0
        
        for d in splits:
            _d = d 
            # If we want to support keeping separator, logic is more complex.
            # For exact parity with LangChain default (keep_separator=False usually for char splitter?),
            # Wait, LangChain RecursiveCharacterTextSplitter defaults:
            # keep_separator=True (It keeps the separator attached to the chunk usually?)
            # Actually LangChain's default behavior is to use `Regex` split which keeps capture groups.
            # But the 'simple' version just splits.

            # Simpler implementation strategy: 
            # 1. Split
            # 2. Check size. If > chunk_size, recurse with `new_separators`
            # 3. If <= chunk_size, accumulate into current buffer (handling overlap/merge)
            
            len_d = self._length_function(_d)
            
            if len_d > self._chunk_size:
                if new_separators:
                    # Recurse
                    sub_chunks = self._split_text(_d, new_separators)
                    good_splits.extend(sub_chunks)
                else:
                    # No more separators, forced to accept it (or hard split if we want)
                    good_splits.append(_d)
            else:
                good_splits.append(_d)
                
        # Now Merge short chunks
        return self._merge_splits(good_splits, separator)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        separator_len = self._length_function(separator)
        
        docs = []
        current_doc = []
        total = 0
        
        for d in splits:
            len_d = self._length_function(d)
            
            if total + len_d + (separator_len if total > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    # Warning: single chunk larger than limit
                    pass
                
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    
                    # Overlap logic
                    # Keep popping from start until it fits?
                    while total > self._chunk_overlap or (total + len_d + separator_len > self._chunk_size and total > 0):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc.pop(0)

            current_doc.append(d)
            total += len_d + (separator_len if len(current_doc) > 1 else 0)
            
        if current_doc:
             doc = self._join_docs(current_doc, separator)
             if doc is not None:
                docs.append(doc)
                
        return docs
        
    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        return text
