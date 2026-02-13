import os
import json
import re
from rag_lib.loaders.structured import StructuredLoader

def main():
    file_path = "C:/Projects/kblib/docs/tasks.docx"
    output_dir = "results/tasks_output_v2" # Separate folder
    os.makedirs(output_dir, exist_ok=True)
    
    # User Configuration V2
    # Start of sections
    # Level 1: ^\d.*класс
    # Level 2: ^Тема.*\d*.*
    
    section_patterns = [
        (1, r"^\d.*класс"),       # Level 1: Class (Shifted up from L2)
        (2, r"^Тема.*\d*.*")      # Level 2: Topic (Shifted up from L3)
    ]
    
    # Level 3: ^\d+\.\d+\.
    # Level 3: Задача.*\d.*
    # We use regex_patterns for these to ensure they split content effectively.
    regex_patterns = [
        (3, r"^\d+\.\d+\."), 
        (3, r"Задача.*\d.*")
    ]
    
    exclusions = [
        r"^Ответ.*:",
        r"^Вопрос.*:",
        r"^Условие.*:",
        r"^Заголовок.*:"
    ]
    
    print(f"Loading {file_path} with V2 configuration...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=False 
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
    
    # 2. Save Markdowns per "Section" 
    # Logic: Since Topic is now Level 2, we create files for Level 2 segments.
    
    sections_dir = os.path.join(output_dir, "sections")
    os.makedirs(sections_dir, exist_ok=True)
    
    current_fhandle = None
    
    for seg in segments:
        if seg.level == 1:
            # Class level. 
            print(f"  Found Class (L1): {seg.metadata.get('title', 'Unknown')}")
            # Optional: Maybe create a folder for the class? 
            # For now, just print.
            
        elif seg.level == 2:
            # Topic (L2) -> New File
            if current_fhandle:
                current_fhandle.close()
            
            title = seg.metadata.get("title", "Untitled").replace(":", "-").replace("/", "-").strip()
            # Truncate title length to avoid path issues
            title = title[:80] 
            # Sanitize filename further
            title = re.sub(r'[<>:"/\\|?*]', '', title)
            
            fname = os.path.join(sections_dir, f"{title}.md")
            current_fhandle = open(fname, 'w', encoding='utf-8')
            print(f"  Creating Section File: {fname}")
            
            current_fhandle.write(f"# {seg.metadata.get('title')}\n\n")
            current_fhandle.write(seg.content + "\n\n")
            
        elif seg.level > 2:
            # Child content (Tasks L3)
            # Append to current open file (L2 Topic)
            if current_fhandle:
                # Use sub-header for L3
                current_fhandle.write(f"### {seg.metadata.get('title')}\n\n")
                current_fhandle.write(seg.content + "\n\n")
        else:
            # L0
            pass
            
    if current_fhandle:
        current_fhandle.close()

if __name__ == "__main__":
    main()
