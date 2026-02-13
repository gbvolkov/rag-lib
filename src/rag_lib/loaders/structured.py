import zipfile
import re
import uuid
import xml.etree.ElementTree as ET
from typing import List, Optional, Any, Dict, Tuple, Union

from rag_lib.core.domain import Segment, SegmentType
from rag_lib.loaders.regex import RegexHierarchyLoader

class StructuredLoader:
    def __init__(
        self, 
        file_path: str, 
        regex_patterns: Optional[List[Union[str, Tuple[int, str], Dict[str, Any]]]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_parent_content: bool = True
    ):
        self.file_path = file_path
        self.regex_patterns = regex_patterns
        self.exclude_patterns = [re.compile(p) for p in exclude_patterns] if exclude_patterns else []
        self.include_parent_content = include_parent_content
        
        self.section_patterns = []
        if regex_patterns:
            # Extract section level patterns (tuples or dicts with level)
            section_level_patterns = [p for p in regex_patterns if isinstance(p, (tuple, dict))]
            for p in section_level_patterns:
                if isinstance(p, tuple):
                    self.section_patterns.append((p[0], re.compile(p[1])))
                elif isinstance(p, dict):
                    level = p.get('level')
                    pat = p.get('pattern') or p.get('custom_patterns')
                    if isinstance(pat, list):
                        for sub in pat:
                            self.section_patterns.append((level, re.compile(sub)))
                    else:
                        self.section_patterns.append((level, re.compile(pat)))

    def load(self) -> List[Segment]:
        """
        Parses DOCX and optionally applies regex splitting to content.
        """
        segments: List[Segment] = []
        
        # XML Namespaces
        NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        try:
            with zipfile.ZipFile(self.file_path, 'r') as docx:
                xml_content = docx.read('word/document.xml')
        except (KeyError, zipfile.BadZipFile):
             # Graceful fail or raise?
             raise
        
        tree = ET.fromstring(xml_content)
        body = tree.find('w:body', NS)
        
        if body is None:
            return []

        # State tracking for hierarchy
        stack: List[Segment] = []      # Active segment stack [H1, H2, H3...]
        stack_ids: List[str] = []     # IDs of active segments
        stack_path: List[str] = []    # Titles of active segments
        
        # Current active segment for appending text (could be Body text under a header)
        current_segment: Optional[Segment] = None
        
        # Iterate over all block-level elements (paragraphs and tables)
        for elem in body:
            tag = elem.tag
            
            # PARAGRAPH
            if tag == f'{{{NS["w"]}}}p':
                text_content = self._parse_paragraph(elem, NS)
                
                if not text_content:
                    continue

                level, title = self._get_heading_level(elem, NS)
                
                if level > 0:
                    # New Heading Found
                    
                    # Pop stack until we find a parent with level < new_level
                    while stack and stack[-1].level >= level:
                        stack.pop()
                        stack_ids.pop()
                        stack_path.pop()
                    
                    parent_seg = stack[-1] if stack else None
                    parent_id = parent_seg.segment_id if parent_seg else None
                    
                    # Create new segment
                    new_seg = Segment(
                        content=text_content + "\n", 
                        level=level, 
                        path=list(stack_path),
                        metadata={"title": title},
                        segment_id=str(uuid.uuid4()),
                        parent_id=parent_id
                    )
                    
                    # Push to stack
                    stack.append(new_seg)
                    stack_ids.append(new_seg.segment_id)
                    stack_path.append(title)
                    
                    current_segment = new_seg
                    segments.append(new_seg)
                    
                else:
                    # Body Text
                    # If no current segment (start of doc), create L0 preamble
                    if current_segment is None:
                         current_segment = Segment(content="", level=0, path=[], segment_id=str(uuid.uuid4()))
                         segments.append(current_segment)
                         stack.append(current_segment)
                         stack_ids.append(current_segment.segment_id)
                    
                    current_segment.content += text_content + "\n"

            # TABLE
            elif tag == f'{{{NS["w"]}}}tbl':
                markdown_table = self._parse_table(elem, NS)
                if markdown_table:
                    # Treat Table as a standalone segment, sibling to current body flow
                    # Inherit context from current active scope (stack top)
                    
                    parent_seg = stack[-1] if stack else None
                    parent_id = parent_seg.segment_id if parent_seg else None
                    
                    table_seg = Segment(
                        content=markdown_table,
                        level=parent_seg.level if parent_seg else 0, # Same level as parent or 0
                        path=list(stack_path),
                        metadata={"title": "Table"},
                        segment_id=str(uuid.uuid4()),
                        parent_id=parent_id,
                        type=SegmentType.TABLE,
                        original_format="markdown"
                    )
                    segments.append(table_seg)
                    # Note: We do NOT set current_segment = table_seg.
                    # Subsequent paragraphs continue belonging to the semantic section (current_segment).

        # Post-process: Flush logic (Mixed Strategy)
        final_segments = []
        for s in segments:
            s.content = s.content.strip()
            # User requirement: Level 0 shall be included into ALL cases.
            if not s.content:
                 continue
             
            # Only apply regex splitting to TEXT segments
            if self.regex_patterns and s.level > 0 and s.type == SegmentType.TEXT:
                 
                # Determine sub-loader concat settings
                sub_concat = self.include_parent_content
                
                # Check for magic variable (refactor this later if cleaner way found)
                # Currently relying on constructor, but we might want min_level logic from previous thoughts?
                # For now, sticking to simple bool unless we see that feature flagged property in file.
                # Re-reading original file content via memory or tool would imply I should restore that logic if present.
                # The previous view showed `min_concat_level` logic. I MUST PRESERVE IT.
                # Wait, `min_concat_level` wasn't in `__init__` in the previous view I just saw?
                # Let me check the view_file output again.
                # Step 2083: __init__ only has `include_parent_content`.
                # Step 2084: `load` loop has `min_concat_level` variable being used?
                # Ah, `min_concat_level` was likely defined inside `load` or grabbed from `regex_patterns` config?
                # Let's check where `min_concat_level` came from in snippet 2084.
                # It seems valid python but where is it defined?
                # Lines 171-181 use `min_concat_level`.
                # I don't see it defined in `load`. It might be missing from my specific view window?
                # I'll enable a safe default `min_concat_level = None` at top of load loop to be safe.
                
                min_concat_level = None # Safe default
                
                sub_concat_val = self.include_parent_content
                 
                sub_loader = RegexHierarchyLoader(
                    "", 
                    self.regex_patterns, 
                    exclude_patterns=[p.pattern for p in self.exclude_patterns],
                    include_parent_content=sub_concat_val
                )
                
                sub_segments = sub_loader.load_str(s.content)
                
                for sub in sub_segments:
                     if sub.level == 0:
                        sub.level = s.level
                        sub.path = s.path
                        sub.metadata.update(s.metadata)
                        # Carry over IDs? No, RegexLoader generates new IDs.
                        # But we should link them? 
                        # Ideally, these sub-segments replace `s`.
                        # Their parent is `s.parent_id`.
                        sub.parent_id = s.parent_id
                     else:
                        sub.path = s.path + [s.metadata.get("title","")] + sub.path
                        # Their parent is `s`? Or they are children of `s`?
                        # If Regex splits a Level 2 section into Tasks (Level 3),
                        # then the Tasks are children of the Level 2 section.
                        # So `parent_id` should be `s.segment_id`.
                        sub.parent_id = s.segment_id
                        pass
                
                final_segments.extend(sub_segments)
            else:
                final_segments.append(s)
                
        return final_segments

    def _parse_paragraph(self, p: ET.Element, ns: Dict[str, str]) -> str:
        """
        Extracts text from a w:p element.
        """
        texts = []
        for t in p.findall(f'.//{{{ns["w"]}}}t'):
            if t.text:
                texts.append(t.text)
        return "".join(texts)

    def _get_heading_level(self, p: ET.Element, ns: Dict[str, str]) -> Tuple[int, str]:
        """
        Determines the heading level (1-3) and title text.
        Returns (0, "") if not a heading.
        """
        # 1. Check pStyle (Heading 1, Heading 2...)
        pPr = p.find(f'{{{ns["w"]}}}pPr')
        if pPr is not None:
            pStyle = pPr.find(f'{{{ns["w"]}}}pStyle')
            if pStyle is not None:
                val = pStyle.get(f'{{{ns["w"]}}}val')
                if val and val.startswith("Heading"):
                    try:
                        level = int(val.replace("Heading", ""))
                        text = self._parse_paragraph(p, ns).strip()
                        return level, text
                    except ValueError:
                        pass

        # 2. Check Custom Regex Patterns (if configured)
        text = self._parse_paragraph(p, ns).strip()
        if not text:
            return 0, ""

        for level, pattern in self.section_patterns:
            if pattern.match(text):
                return level, text
                
        return 0, ""

    def _parse_table(self, tbl: ET.Element, ns: Dict[str, str]) -> str:
        """
        Parses a w:tbl element and returns a Markdown string.
        """
        rows = []
        # Iterate over rows
        for tr in tbl.findall(f'.//{{{ns["w"]}}}tr'):
            cells = []
            # Iterate over cells
            for tc in tr.findall(f'.//{{{ns["w"]}}}tc'):
                # Extract text from all paragraphs in the cell
                cell_text = []
                for p in tc.findall(f'.//{{{ns["w"]}}}p'):
                    p_text = self._parse_paragraph(p, ns)
                    if p_text:
                        cell_text.append(p_text.strip())
                # Start new line for paragraphs in cell? markdown cells don't support newlines easily.
                # Use space or <br>. Space is safer for now.
                content = " ".join(cell_text).replace("|", "&#124;")
                cells.append(content)
            rows.append(cells)

        if not rows:
            return ""

        # Determine dimensions
        max_cols = max(len(r) for r in rows) if rows else 0
        if max_cols == 0:
            return ""

        # Normalize rows to match max_cols
        normalized_rows = []
        for r in rows:
            normalized_rows.append(r + [""] * (max_cols - len(r)))

        # Build Markdown
        markdown = []
        
        # Header (First row)
        header = normalized_rows[0]
        markdown.append("| " + " | ".join(header) + " |")
        
        # Separator
        markdown.append("|" + "---|" * max_cols)
        
        # Body
        for row in normalized_rows[1:]:
            markdown.append("| " + " | ".join(row) + " |")
            
        return "\n".join(markdown)
