import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from rag_lib.processors.raptor import RaptorProcessor
    from rag_lib.raptor.tree_builder import TreeBuilder
    print("Imports successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

print("Syntax check passed.")
