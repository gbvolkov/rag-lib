import zipfile
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Any, Dict, Tuple, Union

from rag_lib.core.domain import Document
from rag_lib.core.logger import logger

class StructuredLoader:
    """
    Parses complex DOCX files into a single Markdown Document.
    Preserves heading hierarchy (#, ##, etc.) and tables.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Parses DOCX to Markdown.
        """
        markdown_content = self._parse_docx()
        if not markdown_content:
            return []
            
        return [Document(page_content=markdown_content, metadata={"source": self.file_path})]
        
    def _parse_docx(self) -> str:
        # XML Namespaces
        NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        try:
            with zipfile.ZipFile(self.file_path, 'r') as docx:
                xml_content = docx.read('word/document.xml')
        except (KeyError, zipfile.BadZipFile, FileNotFoundError) as e:
             logger.error(f"Failed to read DOCX {self.file_path}: {e}")
             return ""
        
        tree = ET.fromstring(xml_content)
        body = tree.find('w:body', NS)
        
        if body is None:
            return ""

        markdown_lines = []
        
        for elem in body:
            tag = elem.tag
            
            # PARAGRAPH
            if tag == f'{{{NS["w"]}}}p':
                text_content = self._parse_paragraph(elem, NS)
                if not text_content.strip():
                    continue

                level, title = self._get_heading_level(elem, NS)
                
                if level > 0:
                    # Append Heading
                    markdown_lines.append(f"{'#' * level} {title}")
                else:
                    # Append Body Text
                    markdown_lines.append(text_content)

            # TABLE
            elif tag == f'{{{NS["w"]}}}tbl':
                markdown_table = self._parse_table(elem, NS)
                if markdown_table:
                    # Separate table with newlines
                    markdown_lines.append("\n" + markdown_table + "\n")

        return "\n\n".join(markdown_lines)

    def _parse_paragraph(self, p: ET.Element, ns: Dict[str, str]) -> str:
        texts = []
        for t in p.findall(f'.//{{{ns["w"]}}}t'):
            if t.text:
                texts.append(t.text)
        return "".join(texts)

    def _get_heading_level(self, p: ET.Element, ns: Dict[str, str]) -> Tuple[int, str]:
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
        return 0, ""

    def _parse_table(self, tbl: ET.Element, ns: Dict[str, str]) -> str:
        rows = []
        for tr in tbl.findall(f'.//{{{ns["w"]}}}tr'):
            cells = []
            for tc in tr.findall(f'.//{{{ns["w"]}}}tc'):
                cell_text = []
                for p in tc.findall(f'.//{{{ns["w"]}}}p'):
                    p_text = self._parse_paragraph(p, ns)
                    if p_text:
                        cell_text.append(p_text.strip())
                content = " ".join(cell_text).replace("|", "&#124;")
                cells.append(content)
            rows.append(cells)

        if not rows:
            return ""

        max_cols = max(len(r) for r in rows) if rows else 0
        if max_cols == 0:
            return ""

        normalized_rows = []
        for r in rows:
            normalized_rows.append(r + [""] * (max_cols - len(r)))

        markdown = []
        header = normalized_rows[0]
        markdown.append("| " + " | ".join(header) + " |")
        markdown.append("|" + "---|" * max_cols)
        for row in normalized_rows[1:]:
            markdown.append("| " + " | ".join(row) + " |")
            
        return "\n".join(markdown)
