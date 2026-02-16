import inspect
import sys
try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    try:
        from langchain.retrievers import MultiVectorRetriever
        print("Imported MultiVectorRetriever from langchain (standard)")
    except ImportError:
        try:
            from langchain.retrievers.multi_vector import MultiVectorRetriever
            print("Imported MultiVectorRetriever from langchain.retrievers.multi_vector")
        except ImportError:
            print("Could not import MultiVectorRetriever")
            sys.exit(1)

try:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
    print("Imported MultiVectorRetriever from langchain.retrievers.multi_vector")
    try:
         print("Fields:", MultiVectorRetriever.model_fields.keys())
    except:
         print("No model_fields using dir:", dir(MultiVectorRetriever))
         
    # Try instantiation
    try:
        MultiVectorRetriever(vectorstore=MockVectorStore(), byte_store=MockStore())
        print("SUCCESS: byte_store accepted")
    except TypeError as e:
        print(f"FAILED: byte_store - {e}")

    try:
        # Check alias?
        MultiVectorRetriever(vectorstore=MockVectorStore(), docstore=MockStore())
        print("SUCCESS: docstore accepted")
    except TypeError as e:
        print(f"FAILED: docstore - {e}")

except ImportError:
    print("Could not import MultiVectorRetriever from anywhere")
