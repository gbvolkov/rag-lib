import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import print_section, setup_environment

from langchain_openai import OpenAIEmbeddings

from rag_lib.chunkers.regex import RegexSplitter
from rag_lib.core.domain import Segment
from rag_lib.loaders.docx import DocXLoader
from rag_lib.retrieval.retrievers import RegexRetriever
from rag_lib.vectors.factory import get_vector_store

"""
E2E Example 06: DOCX Regex Workflow

Features Tested:
1. DocXLoader: Loading DOCX as markdown.
2. RegexSplitter: Splitting text based on custom patterns.
3. RegexRetriever: Retrieving documents that match a regex query.
4. VectorStore: Standard indexing (secondary).
"""


def main() -> None:
    setup_environment()
    print_section("06. DOCX Regex Workflow")

    # 1. Load
    docx_path = Path(__file__).parent.parent / "docs" / "Тестирование (БФТ_Подготовка Банковской гарантии).docx"
    print(f"Loading {docx_path}...")

    loader = DocXLoader(str(docx_path))
    docs = loader.load()
    if not docs:
        print("No content loaded from DOCX.")
        return

    # 2. Chunk (Regex)
    print("Re-chunking using RegexSplitter...")
    full_text = docs[0].page_content

    splitter = RegexSplitter(pattern=r"(Step \d+|Шаг \d+)", chunk_size=500)
    regex_chunks = splitter.split_text(full_text)
    print(f"Regex Splitter produced {len(regex_chunks)} chunks.")

    # 3. Retrieval (RegexRetriever)
    print("Retrieving using RegexRetriever (Pattern Matching)...")

    embeddings = OpenAIEmbeddings()
    _ = get_vector_store("chroma", embeddings, "06_docx_regex")

    segments = [Segment(content=chunk) for chunk in regex_chunks]
    print(f"Loaded {len(segments)} segments.")
    if segments:
        print(f"Sample content: {segments[0].content[:200]}...")

    query = "Chapter 2"
    print(f"Retrieving '{query}'...")

    langchain_docs = [s.to_langchain() for s in segments]
    retriever = RegexRetriever(documents=langchain_docs)
    results = retriever.invoke(query)

    print(f"Regex Pattern Results: {len(results)}")
    for result in results:
        print(f"- {result.page_content[:80]}...")


if __name__ == "__main__":
    main()