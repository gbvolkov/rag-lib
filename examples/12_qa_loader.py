import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.data_loaders import QALoader
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import create_vector_store
from langchain_openai import OpenAIEmbeddings

"""
E2E Example 12: QA Loader Workflow

Features Tested:
1. QALoader: Parsing Q&A format text files.
2. Metadata: Storing 'question' separately from 'answer' content.
3. VectorStore: Indexing.

Expected Results:
- Loading:
    - Input: "docs/interview.txt"
    - Logic: Split by "Q:" and "A:".
    - Output: Segments where content is Answer, metadata has Question.
    - Sample Data: 
        Segment(
            content="I have 5 years experience...", 
            metadata={'question': 'What is your experience?'}
        )
- Indexing:
    - Input: Answer segments.
- Retrieval:
    - Query: "graph database experience"
    - Expected Result: Answer describing graph DB work.
    - Sample Output: "I worked with Neo4j..."
"""

def main():
    setup_environment()
    print_section("12. QA Loader Workflow")

    # 2. Load
    qa_path = Path(__file__).parent.parent / "docs" / "interview.txt"
    print(f"Loading {qa_path}...")
    
    loader = QALoader(str(qa_path))
    segments = loader.load()
    print(f"Loaded {len(segments)} Q&A pairs.")

    # 3. Index
    embeddings = OpenAIEmbeddings()
    vector_store = create_vector_store("chroma", embeddings, "12_qa_loader")
    
    indexer = Indexer(vector_store, embeddings)
    indexer.index(segments)

    # 4. Retrieve
    print("Retrieving Answer...")
    results = vector_store.similarity_search("graph database experience", k=1)
    if results:
        print(f"Answer: {results[0].page_content[:100]}...")

if __name__ == "__main__":
    main()
