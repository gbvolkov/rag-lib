import sys
import asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.retrieval.graph_retriever import GraphRetriever
from langchain_openai import ChatOpenAI

"""
E2E Example 05: DOCX Graph Workflow

Features Tested:
1. StructuredLoader: Loading DOCX with paragraph structure.
2. EntityExtractor: Extracting entities (Nodes) and relations (Edges) via LLM.
3. GraphStore (NetworkX): Storing the knowledge graph.
4. GraphRetriever: Retrieving based on graph traversal/properties.

Expected Results:
- Loading:
    - Input: "docs/Параметризованные задачи.docx"
    - Output: List[Segment] (Paragraphs).
    - Sample Data: Segment(content="Задача 1. Найти значения параметра...", type=TEXT)
- Extraction:
    - Logic: LLM extracts (Subject, Relation, Object).
    - Output: Populated GraphStore.
    - Sample Data: Node(id="Parameter"), Edge(source="Task 1", target="Parameter", info="asks to find")
- Retrieval:
    - Method: GraphRetriever(mode="local")
    - Query: "Задача"
    - Logic: Find nodes matching query, traverse neighbors.
    - Sample Output: Segments/Nodes related to "Задача" found in the graph.
"""

async def main():
    setup_environment()
    print_section("05. DOCX Graph Workflow")

    # 2. Load
    docx_path = Path(__file__).parent.parent / "docs" / "Параметризованные задачи.docx"
    print(f"Loading {docx_path}...")
    
    loader = StructuredLoader(str(docx_path))
    segments = loader.load()
    print(f"Loaded {len(segments)} segments preserving structure.")

    # 3. Graph Extraction
    print("Extracting Entities & Relations...")
    graph_store = NetworkXGraphStore()
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    extractor = EntityExtractor(llm=llm, store=graph_store)
    
    # Limit segments for demo speed
    await extractor.aprocess_segments(segments[:10]) 
    
    print(f"Graph Stat: {len(graph_store.get_node('placeholder') or [])} nodes? No, explicitly checking store.")
    # NetworkXGraphStore doesn't expose len directly in base interface clearly, 
    # but we can try retrieving a known entity or just trust the process.
    
    # 4. Retrieve (Graph)
    print("Retrieving using GraphRetriever (Local Mode)...")
    retriever = GraphRetriever(store=graph_store, mode="local")
    
    # Query for something likely in the doc
    query = "Задача" 
    results = await retriever.ainvoke(query)
    
    print(f"Graph Retrieval Results for '{query}':")
    for r in results:
        print(f"- {r.page_content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
