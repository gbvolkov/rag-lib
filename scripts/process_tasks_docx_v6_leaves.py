import os
import re
from rag_lib.loaders.structured import StructuredLoader

def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '-', name)
    name = re.sub(r'-+', '-', name)
    return name.strip()

def main():
    file_path = "C:/Projects/kblib/docs/tasks.docx"
    output_dir = "results/tasks_output_v6_leaves"
    os.makedirs(output_dir, exist_ok=True)
    
    # Improved Configuration from V4/V5
    section_patterns = [
        (1, r"^\d.*класс"),       
        (2, r"^Тема.*\d*.*"),
        (2, r"^\d+\.\s.*")        # Match "1. Name", "2. Name"
    ]
    
    regex_patterns = [
        (3, r"^\d+\.\d+(?:[–-]\d+\.\d+)?\."), # Match "1.4.", "2.3–2.5."
        (3, r"Задача.*\d.*")      
    ]
    
    exclusions = [
        r"^Ответ.*:",
        r"^Вопрос.*:",
        r"^Условие.*:",
        r"^Заголовок.*:"
    ]
    
    print(f"Loading {file_path} with min_level=1 (Leaf Only Mode)...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=1  # Include parents starting from Level 1 (Class)
    )
    
    all_segments = loader.load()
    print(f"Loaded {len(all_segments)} total segments.")
    
    # Filter for Leaves (Level 3)
    # in this specific taxonomy, only L3 are contents we want to treat as atomic tasks
    leaf_segments = [s for s in all_segments if s.level == 3]
    print(f"Filtered to {len(leaf_segments)} leaf segments (Level 3).")
    
    # Verification of Content
    # Check a sample leaf for L1 content
    sample = next((s for s in leaf_segments if "8 класс" in s.content or "9 класс" in s.content), None)
    if sample:
        print(f"Verification: Sample leaf '{sample.metadata.get('title')}' contains L1 content.")
    else:
        # It might be implicitly present via concatenation logic, let's explicit check
        # loader.load() returns concatenated strings if include_parent_content is set.
        pass

    # Output to files
    print(f"\nWriting segments to {output_dir}...")
    for segment in leaf_segments:
        safe_title = sanitize_filename(segment.metadata.get('title', f'segment_{segment.level}'))
        filename = f"{segment.level}_{safe_title}.md"
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(segment.content)
            
    print("Done writing files.")

if __name__ == "__main__":
    main()
