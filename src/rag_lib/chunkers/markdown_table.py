from typing import List, Callable
import re
from rag_lib.core.domain import Segment, SegmentType

class MarkdownTableSplitter:
    """
    Splits text into Table Segments and Text Segments based on Markdown GFM table syntax.
    """
    def __init__(self):
        # Improved Regex:
        # Blocks starting with a pipe line, followed eventually by a separator line, 
        # and continuing until a non-pipe line.
        
        # NOTE: Python's re engine varies. We want to match:
        # Group 1: The whole table.
        # Structure:
        #   (Line with pipe)+   <- Header
        #   (Line with |-)+     <- Separator
        #   (Line with pipe)*   <- Body
        
        # But GFM is strict: Header | Separator | Body.
        # Let's use a pattern that finds the separator and grabs surrounding pipe-lines.
        
        self.table_pattern = re.compile(
            r'('
            r'(?:^.*?\|.*(?:\n|$))'    # Header Row (any line with pipe, really, but usually text|text)
            r'(?:\s*\|[-:| ]+\|\s*(?:\n|$))' # Separator Row (strict structure)
            r'(?:.*?\|.*(?:\n|$))*'    # Body Rows (any line with pipe)
            r')',
            re.MULTILINE
        )

    def split_text(self, text: str) -> List[Segment]:
        segments = []
        last_pos = 0
        
        for match in self.table_pattern.finditer(text):
            start, end = match.span()
            
            # Text before table
            if start > last_pos:
                text_content = text[last_pos:start].strip()
                if text_content:
                    segments.append(Segment(
                        content=text_content,
                        type=SegmentType.TEXT,
                        original_format="text"
                    ))
            
            # Table Segment
            table_content = match.group(0).strip()
            segments.append(Segment(
                content=table_content,
                type=SegmentType.TABLE,
                original_format="markdown"
            ))
            
            last_pos = end
            
        # Remaining text
        if last_pos < len(text):
            text_content = text[last_pos:].strip()
            if text_content:
                segments.append(Segment(
                    content=text_content,
                    type=SegmentType.TEXT,
                    original_format="text"
                ))
                
        return segments
