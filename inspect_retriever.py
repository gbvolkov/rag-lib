
import inspect
import sys
try:
    from langchain.retrievers import MultiVectorRetriever
    print(f"SUCCESS: Imported MultiVectorRetriever from langchain.retrievers")
    print(f"Signature: {inspect.signature(MultiVectorRetriever.__init__)}")
except ImportError as e:
    print(f"FAIL: langchain.retrievers - {e}")

try:
    from langchain.retrievers import EnsembleRetriever
    print(f"SUCCESS: Imported EnsembleRetriever from langchain.retrievers")
except ImportError as e:
    print(f"FAIL: EnsembleRetriever - {e}")

try:
    from langchain_core.vectorstores import SearchType
    print("SUCCESS: Found SearchType in langchain_core.vectorstores")
except ImportError:
    print("FAIL: SearchType not in langchain_core.vectorstores")

try:
    from langchain.schema import SearchType
    print("SUCCESS: Found SearchType in langchain.schema")
except ImportError:
    print("FAIL: SearchType not in langchain.schema")
