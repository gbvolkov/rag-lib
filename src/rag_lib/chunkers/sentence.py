from typing import List, Callable, Optional
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from rag_lib.chunkers.base import TextSplitter

# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentenceSplitter(TextSplitter):
    """
    Splits text by sentences, respecting chunk size and overlap.
    Falls back to word splitting if a single sentence exceeds chunk size.
    Ported from index-builder.
    """
    def __init__(
        self, 
        chunk_size: int = 4000, 
        chunk_overlap: int = 200, 
        length_function: Callable[[str], int] = len,
        language: str = "russian"
    ):
        super().__init__(chunk_size, chunk_overlap, length_function)
        self.language = language

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks composed of sentences.
        """
        # 1. Tokenize into sentences
        try:
            sentences = sent_tokenize(text, language=self.language)
        except LookupError:
             # Fallback if specific language fails
             sentences = sent_tokenize(text)
             
        # 2. Apply chunking logic (Ported)
        return self._chunk_sentences(sentences)

    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """
        Core logic adapted from index-builder/sentence_splitter.py
        """
        # Preprocess sentences to ensure none is longer than max_chunk_size (minus overlap usually?)
        # Original logic: preprocess against max_chunk_size - overlap_size. 
        # But here overlap is optional. Let's stick to safe max first.
        
        safe_max = self._chunk_size
        
        # Helper for pre-processing
        processed_sentences = []
        for s in sentences:
            if self._length_function(s) > safe_max:
                processed_sentences.extend(self._split_long_sentence(s, safe_max))
            else:
                processed_sentences.append(s)
        
        sentences = processed_sentences
        
        chunks = []
        current_chunk = []
        current_length = 0
        idx = 0
        
        while idx < len(sentences):
            sentence = sentences[idx]
            sentence_length = self._length_function(sentence)
            
            # Check if adding sentence fits
            # Note: we need to account for space separator roughly
            # " ".join(current_chunk + [sentence]) length estimate
            sep_len = 1 if current_chunk else 0
            
            if current_length + sep_len + sentence_length <= self._chunk_size:
                current_chunk.append(sentence)
                current_length += sep_len + sentence_length
                idx += 1
            else:
                if not current_chunk:
                    # Single sentence still too big? Should have been split by preprocess.
                    # Just add it.
                    chunks.append(sentence)
                    idx += 1
                else:
                    # Finalize current
                    chunks.append(" ".join(current_chunk))
                    
                    # Compute Overlap
                    overlap_sentences = []
                    overlap_length = 0
                    
                    # Backtrack from end of current chunk
                    for s in reversed(current_chunk):
                        s_len = self._length_function(s)
                        # Space?
                        s_sep = 1 if overlap_sentences else 0
                        
                        if overlap_length + s_sep + s_len <= self._chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += s_sep + s_len
                        else:
                            break
                    
                    # Start new chunk with overlap
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                    
                    # Don't increment idx, retry current sentence with new (smaller) current_chunk
                    # Wait, original logic attempted to add current sentence to overlap immediately?
                    # Original: if overlap + sentence <= max -> current = overlap + sentence, idx++
                    # Else: current = [], chunks.append(sentence), idx++
                    
                    # Let's align with original logic:
                    sep_new = 1 if current_chunk else 0
                    if current_length + sep_new + sentence_length <= self._chunk_size:
                        current_chunk.append(sentence)
                        current_length += sep_new + sentence_length
                        idx += 1
                    else:
                        # Even with overlap, it doesn't fit?
                        # Flush overlap? No, overlap is just preamble.
                        # Original code: dumps overlap, clears current, emits sentence as own chunk.
                        # This seems aggressive (loses context if sentence is small but just doesn't fit with overlap).
                        
                        # My improvement: if it doesn't fit with overlap, just start clean with sentence?
                        # Or start clean with empty?
                        
                        # Index-Builder Logic:
                        # else: current_chunk = []; chunks.append(sentence); idx+=1
                        
                        # Let's try to start strict new chunk
                        # current_chunk = [sentence]
                        current_chunk = [sentence]
                        current_length = sentence_length
                        idx += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _split_long_sentence(self, sentence: str, max_chunk_size: int) -> List[str]:
        # Simple whitespace fallback for now, or word_tokenize
        words = word_tokenize(sentence, language=self.language)
        return self._accumulate_words(words, max_chunk_size)

    def _accumulate_words(self, words: List[str], max_chunk_size: int) -> List[str]:
        result = []
        current_line = ""
        for word in words:
            # Check word length (recursive split if word > max) - Skipping for brevity unless strict TDD fails
            if self._length_function(word) > max_chunk_size:
                 # Force split hard
                 sub = [word[i:i+max_chunk_size] for i in range(0, len(word), max_chunk_size)]
                 for s in sub:
                     # Calculate candidate length properly
                     sep = 1 if current_line else 0
                     candidate_len = self._length_function(current_line) + sep + self._length_function(s)
                     
                     if candidate_len <= max_chunk_size:
                         current_line += (" " if current_line else "") + s
                     else:
                         if current_line: result.append(current_line)
                         current_line = s
            else:
                 candidate = current_line + (" " if current_line else "") + word
                 if self._length_function(candidate) <= max_chunk_size:
                     current_line = candidate
                 else:
                     if current_line: result.append(current_line)
                     current_line = word
        if current_line:
            result.append(current_line)
        return result
