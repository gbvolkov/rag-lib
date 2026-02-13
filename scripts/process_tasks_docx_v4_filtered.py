import os
import json
import re
from rag_lib.loaders.structured import StructuredLoader

def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', '-', name)
    name = re.sub(r'-+', '-', name)
    return name.strip()

def main():
    file_path = "C:/Projects/kblib/docs/tasks.docx"
    output_dir = "results/tasks_output_v4_filtered"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration V2/V3
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
    
    # Test: Start level 1 (Exclude Level 0)
    # L1 (Class) should NOT have L0 content.
    # L2 (Topic) should HAVE L1 content.
    # L3 (Task) should HAVE L1+L2 content (and NOT L0).
    
    print(f"Loading {file_path} with min_level=1...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=1  # Start inclusion from Level 1
    )
    
    segments = loader.load()
    print(f"Loaded {len(segments)} segments.")
    
    # Check Segments programmatically first
    l0 = next((s for s in segments if s.level==0), None)
    l1 = next((s for s in segments if s.level==1 and "8 класс" in s.metadata.get('title','')), None)
    l2 = next((s for s in segments if s.level==2 and "Тема" in s.metadata.get('title','') and "8 класс" in s.path), None)
    l3 = next((s for s in segments if s.level==3 and "y=x2" in s.metadata.get('title','')), None)
    
    if l0:
        print(f"L0 Content Length: {len(l0.content)}")
        
    if l1:
        print(f"L1 Content Length: {len(l1.content)}")
        # Check if L1 has L0 content (e.g. "Параметризованные задачи")
        if "Параметризованные задачи" in l1.content:
            print("FAIL: L1 contains L0 content!")
        else:
            print("PASS: L1 does NOT contain L0 content.")
            
    if l2:
        print(f"L2 Content Length: {len(l2.content)}")
        # Check if L2 has L1 content ("8 класс")
        if "8 класс" in l2.content:
            print("PASS: L2 contains L1 content.")
        else:
            print("FAIL: L2 missing L1 content!")
            
        # Check if L2 has L0 content
        if "Параметризованные задачи" in l2.content:
             print("FAIL: L2 contains L0 content!")
        else:
             print("PASS: L2 does NOT contain L0 content.")

    if l3:
        print(f"L3 Content Length: {len(l3.content)}")
        # Check if L3 has L1 content
        if "8 класс" in l3.content:
            print("PASS: L3 contains L1 content.")
        else:
            print("FAIL: L3 missing L1 content!")
            
        # Check if L3 has L0 content
        if "Параметризованные задачи" in l3.content:
             print("FAIL: L3 contains L0 content!")
        else:
             print("PASS: L3 does NOT contain L0 content.")

    # Output to files
    print(f"\nWriting segments to {output_dir}...")
    for segment in segments:
        safe_title = sanitize_filename(segment.metadata.get('title', f'segment_{segment.level}'))
        filename = f"{segment.level}_{safe_title}.md"
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(segment.content)
            
    print("Done writing files.")

if __name__ == "__main__":
    main()
