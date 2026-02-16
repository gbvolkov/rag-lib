
import sys
import importlib

def try_import(module, name):
    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, name)
        print(f"SUCCESS: {name} found in {module}")
        return True
    except (ImportError, AttributeError) as e:
        print(f"FAIL: {name} in {module} - {e}")
        return False

print("--- Probing LangChain Imports ---")

# MultiVectorRetriever
try_import("langchain.retrievers", "MultiVectorRetriever")
try_import("langchain.retrievers.multi_vector", "MultiVectorRetriever")
try_import("langchain_community.retrievers", "MultiVectorRetriever")

# EnsembleRetriever
try_import("langchain.retrievers", "EnsembleRetriever")
try_import("langchain.retrievers.ensemble", "EnsembleRetriever")
try_import("langchain_community.retrievers", "EnsembleRetriever")

# SearchType
try_import("langchain.schema", "SearchType")
try_import("langchain_core.vectorstores", "SearchType")
try_import("langchain.vectorstores", "SearchType")
try_import("langchain.retrievers.multi_vector", "SearchType")

print("--- End Probe ---")
