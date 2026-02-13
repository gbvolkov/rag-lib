import json
import os

def main():
    path = r"results\complex_verification\segments.json"
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    for i, seg in enumerate(segments):
        if "Задача 4" in seg.get("metadata", {}).get("title", ""):
            print(f"\n[FOUND] Segment {i}")
            print(f"Title: {seg['metadata']['title']}")
            print(f"Path: {seg.get('path')}")
            print(f"Parent ID: {seg.get('parent_id')}")
            
            # Print previous 5 segments to see context
            start = max(0, i - 5)
            print("\n--- Context (Previous Segments) ---")
            for j in range(start, i):
                prev = segments[j]
                print(f"[{j}] L{prev['level']} Title: {prev.get('metadata', {}).get('title', 'NO TITLE')}")
                print(f"    Path: {prev.get('path')}")
            break

if __name__ == "__main__":
    main()
