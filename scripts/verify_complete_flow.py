import re
import sys
import os
import json
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from rag_lib.loaders.structured import StructuredLoader
from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.core.domain import Segment

def safe_serialize(obj):
    if hasattr(obj, "dict"):
        return obj.dict() # Pydantic v1
    if hasattr(obj, "model_dump"):
        return obj.model_dump() # Pydantic v2
    return str(obj)

def main():
    docx_path = r"C:\Projects\kblib\docs\tasks.docx"
    output_dir = r"results\complex_verification"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- 1. Loading from {docx_path} ---")
    
    # Loader Config
    section_patterns = [
        (1, r"^\d.*класс"),
        (2, r"^Тема.*\d*.*"),
        (3, r"^\d+\.\d+\."),
        (3, r"^Задача.*\d.*")
    ]
    exclusions = [
        r"^Ответ.*:",
        r"^Вопрос.*:",
        r"^Условие.*:",
        r"^Заголовок.*:"
    ]
    
    loader = StructuredLoader(
        file_path=docx_path,
        section_level_patterns=section_patterns,
        exclude_patterns=exclusions,
        include_parent_content=False
    )
    
    segments = loader.load()
    print(f"Loaded {len(segments)} segments.")
    
    # Save Segments JSON
    segments_data = [safe_serialize(s) for s in segments]
    with open(os.path.join(output_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2, ensure_ascii=False)
    
    # Splitter Config
    chunk_pattern = r"(?m)(?=(?:^Ответ.*:|^Вопрос.*:|^Условие.*:|^Заголовок.*:))"
    splitter = RegexSplitter(pattern=chunk_pattern)
    
    print("\n--- 2. Splitting Leaves ---")
    
    all_chunks = []
    report_lines = []
    report_lines.append(f"# Complex Verification Report")
    report_lines.append(f"Date: {datetime.now().isoformat()}")
    report_lines.append(f"Input: {docx_path}")
    report_lines.append(f"Loader Segments: {len(segments)}")
    report_lines.append("\n## Split Statistics\n")
    
    total_chunks = 0
    
    for i, seg in enumerate(segments):
        if not seg.content.strip():
            continue
            
        chunks = splitter.split_segments([seg])
        total_chunks += len(chunks)
        
        for c in chunks:
            all_chunks.append(safe_serialize(c))
            
        # Add detailed sample for the target "Task" segment
        is_task = any("Задача" in p for p in seg.path) or "Задача" in seg.metadata.get("title", "")
        has_structure = "Ответ" in seg.content or "Вопрос" in seg.content
        
        if is_task and has_structure:
             report_lines.append(f"### Sample Verification (Segment {i})")
             report_lines.append(f"- **Path**: {seg.path}")
             report_lines.append(f"- **Content Preview**: `{seg.content[:50].replace(chr(10), ' ')}...`")
             report_lines.append(f"- **Chunks Generated**: {len(chunks)}")
             for j, c in enumerate(chunks):
                 content_preview = c.content.strip().split('\n')[0][:60]
                 report_lines.append(f"  - **Chunk {j}**: `{content_preview}`")
                 report_lines.append(f"    - `parent_id` match: {c.metadata.get('parent_id') == seg.segment_id}")
                 report_lines.append(f"    - `path`: {c.path}")

    # Save Chunks JSON
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    report_lines.append(f"\n## Summary")
    report_lines.append(f"Total Chunks Generated: {total_chunks}")
    
    # Save Report MD
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n--- Complete. Results saved to {output_dir} ---")
    print(f"Total Chunks: {total_chunks}")

if __name__ == "__main__":
    main()
