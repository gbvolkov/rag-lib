import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from example_utils import setup_environment, print_section, save_json_results

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
    - Logic: Use `TextLoader` to load file as Document.
    - Output: List[Document]
- Chunking:
    - Logic: RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)
    - Output: List[Segment] (preserving source metadata)
    - Sample Data: Segment(content="## Einstein Quotes...", metadata={'source': '...'})
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
    - Expected Result: Top match is segment containing Einstein quotes.
    - Sample Output: "Match: ## Einstein Quotes... (Score: 100)"
- JSON Logs (UTF-8) at `docs/results/`: 
    - `02_markdown_enrichment_enriched_segments_yyyymmdd_hhmmss.json`
    - `02_markdown_enrichment_retrieved_results_yyyymmdd_hhmmss.json`
"""

def main():
    setup_environment()
    print_section("02. Markdown Enrichment Workflow")

    # 2. Load
    md_path = Path(__file__).parent.parent / "docs" / "quotes.toscrape.com_index.md"
    print(f"Loading {md_path}...")
    
    from rag_lib.loaders.data_loaders import TextLoader
    loader = TextLoader(str(md_path))
    docs = loader.load()

    # 3. Chunk
    chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    segments = chunker.split_documents(docs)
    
    # Limit for demo speed
    segments = segments[:5]
    save_json_results(segments, "02_markdown_enrichment", "raw_segments")
    
    # 4. Enrich
    print("Enriching first 5 segments...")
    llm = ChatOpenAI(model="gpt-4.1-nano")
    enricher = SegmentEnricher(llm)
    enriched = enricher.enrich(segments)
    
    # Save enriched segments (Preliminary Results)
    save_json_results(enriched, "02_markdown_enrichment", "enriched_segments")
    
    print(f"Enriched Metadata: {enriched[0].metadata}")

    # 5. Retrieve (Fuzzy)
    # FuzzyRetriever now accepts List[Segment] directly and searches their content (which includes enriched metadata).
    print("Retrieving with FuzzyRetriever (searching enriched content)...")
    
    # Pass Segments directly - FuzzyRetriever handles content access and conversion
    # Using 'wratio' for better handling of compound queries (e.g. "einstein quotes")
    retriever = FuzzyRetriever(documents=enriched, threshold=45, mode="wratio") 
    
    query = "einstein" # "einstein" might overlap
    results = retriever.invoke(query)
    
    print(f"Query: {query}")
    for r in results:
        print(f"Match: {r.page_content[:50]}... (Score: {r.metadata.get('fuzzy_score')})")

    # Save retrieval results
    # Convert Documents to dicts for saving if needed, or rely on default serializer in save_json_results
    save_json_results(results, "02_markdown_enrichment", "retrieved_results")

if __name__ == "__main__":
    main()
