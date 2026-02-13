import os
import json
import re
from rag_lib.loaders.structured import StructuredLoader

def main():
    file_path = "C:/Projects/kblib/docs/tasks.docx"
    output_dir = "results/tasks_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # User Configuration
    section_patterns = [
        (2, r"^\d.*класс"),    # Level 2: Class
        (3, r"^Тема.*\d*.*")   # Level 3: Topic
    ]
    
    # "End of sections" treated as Level 4 splitters (Tasks/Subsections)
    # This ensures they are captured as children of the Topic (L3)
    # and don't break the section hierarchy horizontally (unless they are siblings to Topic?)
    # Generally Tasks belong to Topics.
    regex_patterns = [
        (4, r"^\d+\.\d+\."), 
        (4, r"Задача.*\d.*")
    ]
    
    exclusions = [
        r"^Ответ.*:",
        r"^Вопрос.*:",
        r"^Условие.*:",
        r"^Заголовок.*:"
    ]
    
    print(f"Loading {file_path}...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=False # User didn't request this here, but maybe useful? kept False for clean segmentation test
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
    
    # 2. Save Markdowns per "Section" (Level 3 Topics)
    # We will aggregate content for each L3 section + its children.
    # Logic: Iterate segments. If L3, start new file. If L>3, append to current file.
    
    sections_dir = os.path.join(output_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)
    
    current_file = None
    current_fhandle = None
    
    # We might have L2 (Class) as well.
    # Strategy: One file per L3 Segment?
    # Or just dump each segment to a file?
    # User: "one markdown document per section".
    # Assuming "Section" = Level 3 (Topic).
    
    for seg in segments:
        if seg.level == 2:
            # Class level. Maybe just print?
            print(f"  Found Class: {seg.metadata.get('title', 'Unknown')}")
            pass
        elif seg.level == 3:
            # New Topic -> New File
            if current_fhandle:
                current_fhandle.close()
            
            title = seg.metadata.get("title", "Untitled").replace(":", "-").replace("/", "-").strip()
            # Truncate title length
            title = title[:50]
            fname = os.path.join(sections_dir, f"{title}.md")
            current_fhandle = open(fname, 'w', encoding='utf-8')
            print(f"  Creating Section File: {fname}")
            
            current_fhandle.write(f"# {seg.metadata.get('title')}\n\n")
            current_fhandle.write(seg.content + "\n\n")
            
        elif seg.level > 3:
            # Child content (Tasks)
            if current_fhandle:
                current_fhandle.write(f"### {seg.metadata.get('title')}\n\n")
                current_fhandle.write(seg.content + "\n\n")
        else:
            # L0 or other
            pass
            
    if current_fhandle:
        current_fhandle.close()

if __name__ == "__main__":
    main()
