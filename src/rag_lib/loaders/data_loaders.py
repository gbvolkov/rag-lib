import csv
import json
from typing import List, Optional, Union, Dict, Any
from rag_lib.core.domain import Segment

class TableLoader:
    def __init__(self, file_path: str, mode: str = "row", group_by: Optional[str] = None):
        self.file_path = file_path
        self.mode = mode
        self.group_by = group_by

    def load(self) -> List[Segment]:
        segments = []
        try:
            with open(self.file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return []

        if self.mode == "row":
            for i, row in enumerate(rows):
                # Represent row as key-value pairs
                content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                segments.append(Segment(
                    content=content,
                    metadata={"row": i + 1, "source": self.file_path}
                ))
        
        elif self.mode == "group" and self.group_by:
            # Group rows by column value
            groups = {}
            for row in rows:
                key = row.get(self.group_by, "Unknown")
                if key not in groups:
                    groups[key] = []
                groups[key].append(row)
            
            for key, group_rows in groups.items():
                # Serialize group
                content_lines = []
                content_lines.append(f"Group: {self.group_by} = {key}")
                for row in group_rows:
                    row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if k != self.group_by])
                    content_lines.append(row_str)
                
                segments.append(Segment(
                    content="\n".join(content_lines),
                    metadata={"group_key": key, "source": self.file_path}
                ))

        return segments

class JsonLoader:
    def __init__(self, file_path: str, jq_schema: str = "."):
        self.file_path = file_path
        self.jq_schema = jq_schema

    def load(self) -> List[Segment]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
            
        segments = []
        # Simplified JQ: if schema is ".", expect list. if key, access key.
        target = data
        if self.jq_schema != ".":
            # Very basic key access
            keys = self.jq_schema.strip(".").split(".")
            for k in keys:
                if isinstance(target, dict):
                    target = target.get(k, [])
                else:
                    break
        
        if isinstance(target, list):
            for item in target:
                content = json.dumps(item, indent=2) if isinstance(item, (dict, list)) else str(item)
                segments.append(Segment(content=content, metadata={"source": self.file_path}))
        elif isinstance(target, dict):
             # Just one segment?
             segments.append(Segment(content=json.dumps(target, indent=2), metadata={"source": self.file_path}))
             
        return segments

class QALoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Segment]:
        segments = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            return []
            
        # Basic parsing: Split by double newline, look for Q: / A:
        # Or just split by "Q:"
        parts = text.split("Q: ")
        for part in parts:
            if not part.strip(): continue
            
            # part is "What is X?\nA: X is Y."
            lines = part.split("\n", 1)
            question = lines[0].strip()
            
            # Reconstruct content with "Q: " prefix
            content = "Q: " + part.strip()
            segments.append(Segment(
                content=content,
                metadata={"type": "qa", "question": question}
            ))
            
        return segments
