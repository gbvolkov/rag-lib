import json
import os

def main():
    base_dir = r"results\complex_verification"
    chunks_path = os.path.join(base_dir, "chunks.json")
    segments_path = os.path.join(base_dir, "segments.json")
    
    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    print(f"Loaded {len(segments)} segments.")
    seg_ids = {s.get("segment_id") for s in segments}
    
    missing_seg_id = [s for s in segments if not s.get("segment_id")]
    if missing_seg_id:
        print(f"ERROR: {len(missing_seg_id)} segments are missing 'segment_id'!")
    else:
        print("PASS: All segments have 'segment_id'.")
        
    # Inspect a Level 3 segment
    l3_segs = [s for s in segments if s.get("level") == 3]
    if l3_segs:
        print(f"\n[INSPECTION] Found {len(l3_segs)} Level 3 segments.")
        sample = l3_segs[0]
        print("Sample Level 3 Segment:")
        print(f"  ID: {sample.get('segment_id')}")
        print(f"  Parent ID: {sample.get('parent_id')}")
        print(f"  Path: {sample.get('path')}")
        if not sample.get('parent_id'):
             print("  [ERROR] Parent ID is null!")
        else:
             print("  [OK] Parent ID is present.")
    else:
        print("\n[WARNING] No Level 3 segments found to inspect.")
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    missing_parent = [c for c in chunks if not c.get("metadata", {}).get("parent_id")]
    
    # Note: Chunks derived from segments should have parent_id.
    # Are there any chunks that wouldn't? 
    # Only if the source segment didn't have an ID, but we verified they do.
    
    if missing_parent:
        print(f"ERROR: {len(missing_parent)} chunks are missing 'parent_id' in metadata!")
        # Print sample
        print("Sample missing parent chunk:", json.dumps(missing_parent[0], indent=2, ensure_ascii=False)[:200])
    else:
        print("PASS: All chunks have 'parent_id'.")

    # Check Orphan Links
    orphans = [c for c in chunks if c.get("metadata", {}).get("parent_id") not in seg_ids]
    if orphans:
        print(f"WARNING: {len(orphans)} chunks have parent_ids that are not in the segment list (maybe partial load?)")
    else:
        print("PASS: All parent_ids resolve to valid segments.")

if __name__ == "__main__":
    main()
