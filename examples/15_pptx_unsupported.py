import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

"""
E2E Example 15: PPTX Unsupported Workflow

Features Tested:
1. Unsupported Format Handling: Verifying system behavior for known unsupported types.
2. Error Reporting: Ensuring clear error messages for users.

Expected Results:
- Loading:
    - Input: "docs/Digitme Презентация.pptx"
    - Action: Attempt to import/use PPTXLoader.
    - Expected Error: ImportError or "Loader not found".
    - Output to User: "PPTX format is currently NOT SUPPORTED."
    - Sample Data: N/A (Load fails by design).
"""

def main():
    setup_environment()
    print_section("15. PPTX Unsupported Workflow")

    pptx_path = Path(__file__).parent.parent / "docs" / "Digitme Презентация.pptx"
    print(f"Attempting to load {pptx_path}...")
    
    # 1. Check Support
    # Rag-lib currently does NOT have a PPTXLoader.
    # We demonstrate graceful handling or explicit failure.
    
    print("Checking for available PPTX loader...")
    try:
        from rag_lib.loaders.pptx import PPTXLoader
        loader = PPTXLoader(str(pptx_path))
        loader.load()
    except ImportError:
        print(">> ImportError: No 'PPTXLoader' found in rag_lib.loaders.")
        print(">> Result: PPTX format is currently NOT SUPPORTED.")
        print(">> Action: Please convert to PDF or implement PPTXLoader.")

if __name__ == "__main__":
    main()
