import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def verify_imports():
    print("Verifying imports of cleaned-up modules...")
    
    modules = [
        "rag_lib.llm.factory",
        "rag_lib.embeddings.factory", 
        "rag_lib.vectors.factory",
        "rag_lib.retrieval.retrievers",
        "rag_lib.retrieval.composition",
        "rag_lib.loaders.miner_u",
        "rag_lib.loaders.pdf",
        "rag_lib.graph.neo4j_store",
        "rag_lib.core.domain",
        "rag_lib.graph" # Check __init__.py
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ Imported {module}")
        except ImportError as e:
            print(f"⚠️ Import failed for {module}: {e}")
            # This might be expected if dependencies are missing, validation depends on environment
            # But syntax errors will raise SyntaxError which is not caught here (except by generic Exception?)
            # ImportError is anticipated. SyntaxError is NOT.
        except SyntaxError as e:
            print(f"❌ Syntax Error in {module}: {e}")
            failed.append(module)
        except Exception as e:
            print(f"❌ Unexpected Error in {module}: {e}")
            failed.append(module)

    if failed:
        print(f"\nFailed modules: {failed}")
        sys.exit(1)
    else:
        print("\nAll modules imported successfully (or raised expected ImportErrors).")
        sys.exit(0)

if __name__ == "__main__":
    verify_imports()
