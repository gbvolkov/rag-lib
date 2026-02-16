import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section

# 1. Imports
from rag_lib.core.domain import Segment
from rag_lib.chunkers.recursive import RecursiveCharacterTextSplitter
from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.retrieval.retrievers import FuzzyRetriever
from langchain_openai import ChatOpenAI

"""
E2E Example 02: Markdown Enrichment Workflow

Features Tested:
1. Markdown Loading: Reading markdown files.
2. RecursiveSplitter: Chunking with structure preservation.
3. SegmentEnricher: LLM-based metadata extraction (Title, Keywords, Summary).
4. FuzzyRetriever: Approximate string matching on segments.

Expected Results:
- Loading:
    - Input: "docs/quotes.toscrape.com_index.md"
    - Output: Raw markdown string.
- Chunking:
    - Logic: RecursiveCharacterTextSplitter(chunk_size=1000)
    - Output: List[str] -> Wrapped in List[Segment]
    - Sample Data: Segment(content="## Einstein Quotes...", original_format="markdown")
- Enrichment:
    - Input: Segments
    - Logic: GPT-3.5 Extract -> Populate Metadata
    - Output: Segment with enriched metadata.
    - Sample Data: 
        Segment(
            metadata={
                'title': 'Einstein Quotes',
                'keywords': ['physics', 'science'],
                'summary': 'Collection of quotes by Albert Einstein.'
            }
        )
- Retrieval:
    - Query: "einstein quotes"
    - Method: Fuzzy partial ratio match.
    - Expected Result: Top match is segment containing Einstein quotes.
    - Sample Output: "Match: ## Einstein Quotes... (Score: 100)"
"""

def main():
    setup_environment()
    print_section("02. Markdown Enrichment Workflow")

    # 2. Load
    md_path = Path(__file__).parent.parent / "docs" / "quotes.toscrape.com_index.md"
    print(f"Loading {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. Chunk
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = chunker.split_text(text)
    segments = [Segment(content=c, original_format="markdown", segment_id=f"md_{i}") for i, c in enumerate(chunks)][:5] # Limit for demo speed

    # 4. Enrich
    print("Enriching first 5 segments...")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    enricher = SegmentEnricher(llm)
    enriched = enricher.enrich(segments)
    
    print(f"Enriched Metadata: {enriched[0].metadata}")

    # 5. Retrieve (Fuzzy - No Vector Store needed for this demo of 'FuzzyRetriever' class logic)
    # Note: FuzzyRetriever usually works on a list of strings or documents.
    # We will demonstrate it in-memory.
    
    print("Retrieving with FuzzyRetriever...")
    # Mocking a simple retrieval over these segments
    docs = [s.to_langchain() for s in enriched]
    retriever = FuzzyRetriever(documents=docs, threshold=50) 
    # FuzzyRetriever expects a list of Segments/Docs and does fuzzy match on content
    
    query = "einstein quotes" # "einstein" might overlap
    results = retriever.invoke(query)
    
    print(f"Query: {query}")
    for r in results:
        print(f"Match: {r.page_content[:50]}... (Score: {r.metadata.get('fuzzy_score')})")

if __name__ == "__main__":
    main()
