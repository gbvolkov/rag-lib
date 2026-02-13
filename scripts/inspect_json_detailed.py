import json

def main():
    with open("results/tasks_output_v3_concat/all_segments.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find L1 "8 класс"
    l1 = next((s for s in data if s['level'] == 1 and "8" in s['metadata'].get('title', '')), None)
    if l1:
        print(f"L1 Title: {l1['metadata'].get('title')}")
        print(f"L1 Content Length: {len(l1['content'])}")
        print(f"L1 Start: {repr(l1['content'][:50])}")
        print(f"L1 End: {repr(l1['content'][-50:])}")
    else:
        print("L1 not found")

    # Find L2 under L1
    l2 = next((s for s in data if s['level'] == 2 and l1 and s['path'] == [l1['metadata']['title']] and "12" in s['metadata'].get('title', '')), None)
    if l2:
        print(f"\nL2 Title: {l2['metadata'].get('title')}")
        print(f"L2 Path: {l2['path']}")
        print(f"L2 Content Length: {len(l2['content'])}")
        print(f"L2 Start: {repr(l2['content'][:50])}")
    else:
        print("L2 not found")
        
    # Find L3 under L2
    if l2:
        target_path = l2['path'] + [l2['metadata']['title']]
        l3 = next((s for s in data if s['level'] == 3 and s['path'] == target_path and "y=x2" in s['metadata'].get('title', '')), None)
        if l3:
            print(f"\nL3 Title: {l3['metadata'].get('title')}")
            print(f"L3 Content Length: {len(l3['content'])}")
            print(f"L3 Start: {repr(l3['content'][:50])}")
        else:
            print("L3 not found")

if __name__ == "__main__":
    main()
