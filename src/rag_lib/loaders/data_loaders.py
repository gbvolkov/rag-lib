import csv
import json
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple
from rag_lib.core.domain import Document


class SchemaDialect(str, Enum):
    DOT_PATH = "dot_path"


def _normalize_schema(schema: str) -> str:
    if schema is None:
        return "."
    normalized = str(schema).strip()
    return normalized if normalized else "."


def select_schema_target(
    data: Any,
    schema: str = ".",
    schema_dialect: SchemaDialect = SchemaDialect.DOT_PATH,
) -> Tuple[bool, Any]:
    """
    Select nested JSON target using the configured schema dialect.
    Returns (False, None) when the path cannot be resolved.
    """
    if schema_dialect != SchemaDialect.DOT_PATH:
        raise ValueError(f"Unsupported schema_dialect: {schema_dialect}")

    normalized_schema = _normalize_schema(schema)
    if normalized_schema == ".":
        return True, data

    path = normalized_schema[1:] if normalized_schema.startswith(".") else normalized_schema
    if not path:
        return True, data

    target = data
    for token in (part for part in path.split(".") if part):
        if isinstance(target, dict):
            if token not in target:
                return False, None
            target = target[token]
            continue

        if isinstance(target, list):
            try:
                index = int(token)
            except ValueError:
                return False, None
            if index < 0 or index >= len(target):
                return False, None
            target = target[index]
            continue

        return False, None

    return True, target


def iter_json_leaf_paths(
    value: Any,
    path: Tuple[str, ...] = (),
) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    """
    Iterate over scalar leaf values of JSON-like structures.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_json_leaf_paths(child, path + (str(key),))
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_json_leaf_paths(child, path + (str(index),))
        return

    yield path, value


def _format_markdown_path(path: Tuple[str, ...]) -> str:
    if not path:
        return "value"

    parts: List[str] = []
    for token in path:
        if token.isdigit():
            if not parts:
                parts.append(f"[{token}]")
            else:
                parts[-1] = f"{parts[-1]}[{token}]"
        else:
            parts.append(token)
    return ".".join(parts)


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    return str(value)


def _render_markdown_item(item: Any, index: int) -> str:
    lines = [f"## Item {index}"]
    leaf_items = list(iter_json_leaf_paths(item))
    if not leaf_items:
        lines.append(f"- value: {_format_markdown_value(item)}")
        return "\n".join(lines)

    for path, value in leaf_items:
        lines.append(f"- {_format_markdown_path(path)}: {_format_markdown_value(value)}")
    return "\n".join(lines)


def render_json_as_markdown(value: Any) -> str:
    """
    Render selected JSON payload into a readable markdown representation.
    """
    if isinstance(value, list):
        if not value:
            return "## Item 1\n- value: []"
        return "\n\n".join(_render_markdown_item(item, i) for i, item in enumerate(value, start=1))
    return _render_markdown_item(value, 1)


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
    def __init__(
        self,
        file_path: str,
        output_format: Literal["json", "markdown"] = "json",
        schema: str = ".",
        schema_dialect: SchemaDialect = SchemaDialect.DOT_PATH,
        ensure_ascii: bool = False,
    ):
        if output_format not in ("json", "markdown"):
            raise ValueError("output_format must be either 'json' or 'markdown'.")

        if isinstance(schema_dialect, str):
            schema_dialect = SchemaDialect(schema_dialect)

        self.file_path = file_path
        self.output_format = output_format
        self.schema = _normalize_schema(schema)
        self.schema_dialect = schema_dialect
        self.ensure_ascii = ensure_ascii

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                parsed = json.loads(content)

            found, target = select_schema_target(parsed, self.schema, self.schema_dialect)
            if not found:
                return []

            if self.output_format == "json":
                if self.schema == ".":
                    rendered = content
                else:
                    rendered = json.dumps(target, ensure_ascii=self.ensure_ascii)
            else:
                rendered = render_json_as_markdown(target)

            metadata = {
                "source": self.file_path,
                "source_type": "json",
                "output_format": self.output_format,
                "schema": self.schema,
                "schema_dialect": self.schema_dialect.value,
            }
            return [Document(page_content=rendered, metadata=metadata)]
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
