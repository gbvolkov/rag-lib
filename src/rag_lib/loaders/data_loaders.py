import csv
import json
from typing import List, Optional, Union, Dict, Any
from rag_lib.core.domain import Document, Segment

class TableLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                # Read as CSV to handle headers
                reader = csv.reader(f)
                rows = list(reader)
                if not rows:
                     return []
                
                # Convert to Markdown Table format to preserve structure for splitters
                header = rows[0]
                md_lines = []
                
                # Header row
                md_lines.append("| " + " | ".join(header) + " |")
                # Separator row
                md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                # Data rows
                for row in rows[1:]:
                     md_lines.append("| " + " | ".join(row) + " |")
                
                content = "\n".join(md_lines)
                
                return [Document(page_content=content, metadata={"source": self.file_path})]
        except Exception:
             return []

class JsonLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                # Load the entire file content as one document
                f.seek(0)
                content = f.read()
                # Validate JSON? Optional.
                json.loads(content) # Just to check validity
                return [Document(page_content=content, metadata={"source": self.file_path})]
        except Exception:
            return []

class QALoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": self.file_path})]
        except Exception:
            return []

class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]: 
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": self.file_path})]
        except Exception:
            return []
