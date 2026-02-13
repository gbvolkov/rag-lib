from typing import List, Tuple, Optional, Union, Dict
import re
from rag_lib.core.domain import Segment

class RegexHierarchyLoader:
    """
    Splits text into Segments based on hierarchy patterns.
    Supports multiple regexes per level, exclusion patterns, and optional leaf content concatenation.
    """
    def __init__(
        self, 
        file_path: str, 
        patterns: Union[List[Tuple[int, str]], List[Dict]], 
        exclude_patterns: Optional[List[str]] = None,
        include_parent_content: Union[bool, int] = False
    ):
        """
        :param patterns: List of (level, regex) OR List of dicts {'level': int, 'pattern': str|list}.
        :param exclude_patterns: List of regex strings to ignore (e.g. comments).
        :param include_parent_content: If True (or 0), prefixes ancestor content to each segment. 
                                       If Int > 0, only includes ancestors with level >= Int.
        """
        self.file_path = file_path
        self.exclude_patterns = [re.compile(p) for p in exclude_patterns] if exclude_patterns else []
        self.include_parent_content = include_parent_content
        
        # Normalize patterns to List[(level, re.Pattern)]
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

    def load(self) -> List[Segment]:
        """
        Reads file and returns list of Segments.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
             raise
             
        return self.load_str(text)

    def load_str(self, text: str) -> List[Segment]:
        segments: List[Segment] = []
        
        lines = text.splitlines(keepends=True)

        if not lines:
            return []

        # Pass 1: Tag lines with (Level, Title) or None
        tagged_lines = []
        for line in lines:
            line_str = line.rstrip('\n')
            
            # Exclude?
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
            
        # Pass 2: Build Segments with Concatenation
        
        # Determine strict min level
        min_concat_level = None
        if isinstance(self.include_parent_content, bool):
            if self.include_parent_content:
                min_concat_level = 0
        elif isinstance(self.include_parent_content, int):
            min_concat_level = self.include_parent_content
            
        # Root segment (Level 0) is always present.
        root = Segment(content="", level=0, path=[], metadata={"title": "ROOT"})
        segments.append(root)
        stack: List[Segment] = [root] # Stack of active segments

        for line, match in tagged_lines:
            if match:
                level, title = match
                
                # Manage Stack: Pop segments that are at the same or higher level
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                    
                parent_seg = stack[-1] # The new parent after popping
                
                # Path construction
                path = []
                if parent_seg.level > 0:
                    # Path is parent's path + parent's title
                    path = parent_seg.path + [parent_seg.metadata["title"]]
                    
                # Create new Segment
                new_seg = Segment(content=line, level=level, path=path, metadata={"title": title})
                
                # Leaf Concatenation
                if min_concat_level is not None:
                     if parent_seg.level >= min_concat_level:
                         new_seg.content = parent_seg.content + line
                     else:
                         new_seg.content = line # Break chain logic? 
                         # If parent is below threshold, we start fresh.
                         # Note: parent_seg.content might NOT include its own parent if it was also below threshold.
                         # This consistent logic builds chains from the first valid ancestor.
                
                segments.append(new_seg)
                stack.append(new_seg)
                
            else:
                # Ordinary line. Append to current active segment (top of stack)
                current_seg = stack[-1]
                current_seg.content += line
        
        # Post-Processing: Strip content
        final_segments = []
        for s in segments:
            s.content = s.content.strip()
            if s.content:
                 final_segments.append(s)
            
        # If only the root segment existed and it was empty, keep it if it's the only one (maybe?)
        # Actually R-04 "Just text" means Root has content.
        # If input is empty string lines=[], returns [].
        
        return final_segments
