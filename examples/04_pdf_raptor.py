import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, load_pdf_text

# 1. Imports
from rag_lib.processors.raptor import RaptorProcessor
from rag_lib.core.indexer import Indexer
from rag_lib.vectors.factory import get_vector_store
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_lib.core.domain import Segment

"""
E2E Example 04: PDF RAPTOR Workflow

Features Tested:
1. PDFLoader: Text extraction.
2. RecursiveSplitter: Initial leaf generation.
3. RaptorProcessor: Hierarchical Clustering & Summarization Tree.
4. VectorStore: Indexing tree segments (leaves + summaries).
5. Hierarchical Retrieval: Querying the augmented tree.

Expected Results:
- Loading & Splitting:
    - Input: "docs/Georgy Volkov ru.pdf"
    - Output: Leaf Segments (~10-20 small chunks).
    - Sample Data: Segment(content="Georgy Volkov, Backend Developer...", segment_id="leaf_0")
- RAPTOR Processing:
    - Logic: Embed -> Cluster (GMM/UMAP) -> Summarize (LLM) -> Repeat.
    - Output: List[Segment] containing Leaves (Level 0) and Summaries (Level 1+).
    - Sample Data: 
        Segment(
            content="Summary of clusters regarding experience...", 
            metadata={'level': 1, 'children': ['leaf_0', 'leaf_1']}
        )
- Indexing:
    - Input: Full Tree (Leaves + Summaries).
    - Output: All levels indexed.
- Retrieval:
    - Query: "backend developer experience"
    - Expected Result: Matches from both detailed leaves and high-level summaries.
"""

async def main():
    setup_environment()
    print_section("04. PDF RAPTOR Workflow")
    
    # 2. Load
    pdf_path = Path(__file__).parent.parent / "docs" / "Georgy Volkov ru.pdf"
    print(f"Loading {pdf_path}...")
    
    # Use simple Pypdf loader from utils for raw text to feed RAPTOR
    segment = load_pdf_text(pdf_path)
    if not segment:
        print("Failed to load PDF.")
        return
    raw_text = segment.content
    
    # Initial split to leaves (sentences or small chunks)
    # Raptor needs many small leaves to cluster.
    from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    leaf_texts = splitter.split_text(raw_text)
    
    leaf_segments = [Segment(content=t, segment_id=f"leaf_{i}") for i, t in enumerate(leaf_texts)]
    print(f"Generated {len(leaf_segments)} leaf segments.")

    # 3. RAPTOR Processing
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    
    print("Initializing RAPTOR Processor...")
    processor = RaptorProcessor(
        llm=llm, 
        embeddings=embeddings, 
        max_levels=3
        # threshold=0.5 # GMM Threshold - Not supported in RaptorProcessor init currently
    )
    
    # This might take time (Clustering + Summarizing)
    print("Running RAPTOR (Hierarchical Summarization)...")
    try:
        raptor_tree = await processor.aprocess_segments(leaf_segments)
        print(f"RAPTOR built tree with {len(raptor_tree)} total segments (Leaves + Summaries).")
    except ImportError as e:
        print(f"RAPTOR Dependencies missing: {e}")
        return

    # 4. Index
    vector_store = get_vector_store("chroma", embeddings, "04_pdf_raptor")
    indexer = Indexer(vector_store, embeddings)
    await indexer.aindex(raptor_tree)

    # 5. Retrieve (Hierarchical)
    # We can query specific levels or the whole tree
    print("Retrieving from RAPTOR Tree...")
    results = vector_store.similarity_search("backend developer experience", k=3)
    
    for r in results:
        level = r.metadata.get("level", -1)
        print(f"[Level {level}] {r.page_content[:150]}...")

if __name__ == "__main__":
    asyncio.run(main())
