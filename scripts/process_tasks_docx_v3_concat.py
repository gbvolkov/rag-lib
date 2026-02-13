import os
import json
import re
from rag_lib.loaders.structured import StructuredLoader

def sanitize_filename(name):
    # Remove invalid chars
    name = re.sub(r'[<>:"/\\|?*]', '-', name)
    # Collapse dashes
    name = re.sub(r'-+', '-', name)
    return name.strip()

def main():
    file_path = "C:/Projects/kblib/docs/tasks.docx"
    output_dir = "results/tasks_output_v3_concat"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration V2/V3 (Same patterns)
    section_patterns = [
        (1, r"^\d.*класс"),       # Level 1
        (2, r"^Тема.*\d*.*")      # Level 2
    ]
    
    regex_patterns = [
        (3, r"^\d+\.\d+\."),      # Level 3
        (3, r"Задача.*\d.*")      # Level 3
    ]
    
    exclusions = [
        r"^Ответ.*:",
        r"^Вопрос.*:",
        r"^Условие.*:",
        r"^Заголовок.*:"
    ]
    
    print(f"Loading {file_path} with V3 configuration (One file per segment) + PARENT CONTENT...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=True  # ENABLED
    )
    
    segments = loader.load()
    print(f"Loaded {len(segments)} segments.")
    
    # 1. Save JSON
    json_path = os.path.join(output_dir, "all_segments.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json_data = [
            {
                "level": s.level,
                "path": s.path,
                "content": s.content,
                "metadata": s.metadata
            }
            for s in segments
        ]
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {json_path}")
    
    # 2. Save Markdowns per "Section" (EVERY Segment > 0)
    sections_dir = os.path.join(output_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)
    
    for seg in segments:
        if seg.level > 0:
            # Construct a unique filename based on path + title
            components = seg.path + [seg.metadata.get("title", f"Untitled_L{seg.level}")]
            safe_components = [sanitize_filename(c)[:30] for c in components] 
            
            filename = " - ".join(safe_components) + ".md"
            full_path = os.path.join(sections_dir, filename)
            
            counter = 1
            while os.path.exists(full_path):
                filename = " - ".join(safe_components) + f" ({counter}).md"
                full_path = os.path.join(sections_dir, filename)
                counter += 1
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(f"# {seg.metadata.get('title')}\n\n")
                f.write(f"**Path**: {' > '.join(seg.path)}\n")
                f.write(f"**Level**: {seg.level}\n\n")
                f.write("---\n\n")
                f.write(seg.content)
            
    print(f"Saved {len([s for s in segments if s.level > 0])} separate markdown files to {sections_dir}")

if __name__ == "__main__":
    main()
