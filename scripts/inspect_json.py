import json

def main():
    with open("results/tasks_output_v3_concat/all_segments.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Total Segments: {len(data)}")
    
    # 1. Check L1 Segment (Class 8)
    l1 = next((s for s in data if "8 класс" in s['metadata'].get('title', '')), None)
    if l1:
        print("\n--- Level 1 Content ---")
        print(f"Title: {l1['metadata'].get('title')}")
        print(f"Content Start: {l1['content'][:100]}")
    
    # 2. Check L2 Segment (Topic 12)
    l2 = next((s for s in data if "Тема" in s['metadata'].get('title', '') and "12" in s['metadata'].get('title', '') and "8 класс" in s['path']), None)
    # Check strict path or title
    if l2:
        print("\n--- Level 2 Content ---")
        print(f"Title: {l2['metadata'].get('title')}")
        print(f"Path: {l2['path']}")
        print(f"Content Start: {l2['content'][:200]}")
        
    # 3. Check L3 Segment (y=x2)
    l3 = next((s for s in data if "y=x2" in s['metadata'].get('title', '')), None)
    if l3:
        print("\n--- Level 3 Content ---")
        print(f"Title: {l3['metadata'].get('title')}")
        print(f"Path: {l3['path']}")
        print(f"Content Start: {l3['content'][:300]}")

if __name__ == "__main__":
    main()
