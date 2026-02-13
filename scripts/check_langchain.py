try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("langchain.text_splitter found")
except ImportError as e:
    print(f"langchain.text_splitter failed: {e}")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("langchain_text_splitters found")
except ImportError as e:
    print(f"langchain_text_splitters failed: {e}")
