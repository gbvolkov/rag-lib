from typing import List, Tuple, Optional, Union, Dict, Any, Iterable
import re
import uuid
from rag_lib.core.domain import Segment, SegmentType, Document
from rag_lib.chunkers.base import TextSplitter

class RegexHierarchySplitter(TextSplitter):
    """
    Splits text into Segments based on hierarchy patterns.
    """
    def __init__(
        self, 
        patterns: Union[List[Tuple[int, str]], List[Dict]], 
        exclude_patterns: Optional[List[str]] = None,
        include_parent_content: Union[bool, int] = False
    ):
        """
        :param patterns: List of (level, regex) OR List of dicts {'level': int, 'pattern': str|list}.
        :param exclude_patterns: List of regex strings to ignore (e.g. comments).
        :param include_parent_content: If True (or 0), prefixes ancestor content to each segment. 
        """
        super().__init__()
        self.exclude_patterns = [re.compile(p) for p in exclude_patterns] if exclude_patterns else []
        self.include_parent_content = include_parent_content
        
        # Normalize patterns
        self.compiled_patterns = []
        for p in patterns:
            if isinstance(p, tuple):
                level, pat = p
                self.compiled_patterns.append((level, re.compile(pat)))
            elif isinstance(p, dict):
                level = p.get('level')
                pat = p.get('pattern') or p.get('custom_patterns')
                if isinstance(pat, list):
                    for sub_pat in pat:
                        self.compiled_patterns.append((level, re.compile(sub_pat)))
                else:
                    self.compiled_patterns.append((level, re.compile(pat)))

    def split_text(self, text: str) -> List[str]:
        """
        Required by Base. Logic here is complex hierarchy. 
        We delegate to create_segments to get full objects, then return content?
        Or implement the splitting logic here returning strings?
        Hierarchy splitter *needs* to return Segments to preserve the tree structure.
        Base `split_text` returns `List[str]`. 
        If we return strings, we lose hierarchy metadata (parent_id, path).
        So we should rely on `create_segments` overriding.
        """
        # Return naive split by newlines as fallback check?
        return text.splitlines()

    def create_segments(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Segment]:
        """
        Override to implement hierarchical splitting.
        """
        if metadata is None:
            metadata = {}
            
        segments: List[Segment] = []
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        # Pass 1: Tag lines
        tagged_lines = []
        for line in lines:
            line_str = line.rstrip('\n')
            is_excluded = False
            for excl in self.exclude_patterns:
                if excl.search(line_str):
                    is_excluded = True
                    break
            
            match_data = None
            if not is_excluded:
                for level, pattern in self.compiled_patterns:
                    m = pattern.search(line_str)
                    if m:
                        title = m.group(1).strip() if m.groups() else line_str[m.end():].strip()
                        if not title: title = line_str.strip()
                        match_data = (level, title)
                        break
            tagged_lines.append((line, match_data))

        # Pass 2: Build Segments
        min_concat_level = None
        if isinstance(self.include_parent_content, bool):
             if self.include_parent_content: min_concat_level = 0
        elif isinstance(self.include_parent_content, int):
             min_concat_level = self.include_parent_content

        # Root segment
        root_meta = metadata.copy()
        root_meta["title"] = "ROOT"
        root = Segment(content="", level=0, path=[], metadata=root_meta, segment_id=str(uuid.uuid4()))
        segments.append(root)
        stack: List[Segment] = [root]

        for line, match in tagged_lines:
            if match:
                level, title = match
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                
                parent_seg = stack[-1]
                path = []
                if parent_seg.level > 0:
                    path = parent_seg.path + [parent_seg.metadata.get("title", "")]
                
                meta = metadata.copy()
                meta["title"] = title
                meta["split_strategy"] = self.__class__.__name__

                # Parent ID logic
                parent_id = parent_seg.segment_id
                # If parent is Root (level 0) and has no content, treat as top-level (no parent)
                if parent_seg.level == 0 and not parent_seg.content.strip():
                    parent_id = None
                
                new_seg = Segment(
                    content=line, 
                    level=level, 
                    path=path, 
                    metadata=meta,
                    segment_id=str(uuid.uuid4()),
                    parent_id=parent_id
                )
                
                # Leaf Concatenation logic
                if min_concat_level is not None:
                     if parent_seg.level >= min_concat_level:
                         new_seg.content = parent_seg.content + line
                     # else start fresh (default)
                
                segments.append(new_seg)
                stack.append(new_seg)
            else:
                current_seg = stack[-1]
                current_seg.content += line

        final_segments = []
        for s in segments:
            # Strip content
            s.content = s.content.strip()
            # If it's the Root segment and empty, skip it
            if s.level == 0 and not s.content:
                continue
            
            if s.content:
                final_segments.append(s)

        return final_segments
