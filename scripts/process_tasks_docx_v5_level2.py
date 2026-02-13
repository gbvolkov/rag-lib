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
    output_dir = "results/tasks_output_v5_level2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
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
    
    # Test: Start level 2
    # L1 (Class) should NOT have L0.
    # L2 (Topic) should NOT have L1.
    # L3 (Task) should HAVE L2 content, but NOT L1.
    
    print(f"Loading {file_path} with min_level=2...")
    loader = StructuredLoader(
        file_path,
        section_level_patterns=section_patterns,
        regex_patterns=regex_patterns,
        exclude_patterns=exclusions,
        include_parent_content=2  # Start inclusion from Level 2
    )
    
    segments = loader.load()
    print(f"Loaded {len(segments)} segments.")
    
    # Find specific segments to verify
    l0 = next((s for s in segments if s.level==0), None)
    l1 = next((s for s in segments if s.level==1 and "8 класс" in s.metadata.get('title','')), None)
    l2 = next((s for s in segments if s.level==2 and "Тема" in s.metadata.get('title','') and "8 класс" in s.path), None)
    l3 = next((s for s in segments if s.level==3 and "y=x2" in s.metadata.get('title','')), None)
    
    # Verify L1
    if l1:
        print(f"\n--- Checking Level 1 ({l1.metadata.get('title')}) ---")
        if len(l1.content) < 1000: # L0 is ~7k chars
             print("PASS: L1 does NOT contain L0 content.")
        else:
             print("FAIL: L1 contains L0 content!")

    # Verify L2
    if l2:
        print(f"\n--- Checking Level 2 ({l2.metadata.get('title')}) ---")
        # Check for L1 content ("8 класс")
        if "8 класс" not in l2.content:
            print("PASS: L2 does NOT contain L1 content.")
        else:
            print("FAIL: L2 contains L1 content!")
            
    # Verify L3
    if l3:
        print(f"\n--- Checking Level 3 ({l3.metadata.get('title')}) ---")
        # Check for L1 content
        if "8 класс" not in l3.content:
            print("PASS: L3 does NOT contain L1 content.")
        else:
            print("FAIL: L3 contains L1 content!")
            
        # Check for L2 content
        if "Тема" in l3.content:
             print("PASS: L3 contains L2 content.")
        else:
             print("FAIL: L3 missing L2 content!")

    # Output to files
    print(f"\nWriting segments to {output_dir}...")
    for segment in segments:
        safe_title = sanitize_filename(segment.metadata.get('title', f'segment_{segment.level}'))
        # Create hierarchy-based filename or just flat? V3 used flat for unique segments.
        # Let's use a flat structure with level prefix for easy sorting/viewing
        filename = f"{segment.level}_{safe_title}.md"
        
        # Handle duplicates if any (though titles should be unique-ish)
        # For this test, simpler is better.
        
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(segment.content)
            
    print("Done writing files.")

if __name__ == "__main__":
    main()
