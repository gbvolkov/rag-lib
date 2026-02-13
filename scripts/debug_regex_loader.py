from rag_lib.loaders.regex import RegexHierarchyLoader

def main():
    content = "# Ch1\nText1\n## Sec1\nText2"
    loader = RegexHierarchyLoader("dummy", patterns=[(1, r"^# "), (2, r"^## ")])
    # We mock load call by calling load_str directly if possible, or patching open
    # But RegexHierarchyLoader has load_str.
    
    segments = loader.load_str(content)
    
    print(f"Total Segments: {len(segments)}")
    for i, s in enumerate(segments):
        print(f"--- Segment {i} ---")
        print(f"Level: {s.level}")
        print(f"Title: {s.metadata.get('title')}")
        print(f"Content: {repr(s.content)}")
        print(f"Strip Content: {repr(s.content.strip())}")

if __name__ == "__main__":
    main()
