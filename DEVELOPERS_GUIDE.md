# rag-lib Developer's Guide

## 1. Overview

`rag-lib` is a specialized Python library designed for **Advanced RAG (Retrieval-Augmented Generation)** pipelines. Unlike generic RAG tools, `rag-lib` focuses on **hierarchical segmentation** and **layout-aware parsing**, ensuring that complex documents (PDFs with tables, DOCX with strict headers, huge CSVs) are preserved as meaningful "Segments" rather than arbitrary text chunks.

### Key Philosophy: The "Segment"

The atomic unit of `rag-lib` is the **Segment** (`rag_lib.core.domain.Segment`).

- **Not just text**: A Segment can be a `TABLE`, `IMAGE`, `AUDIO`, or `TEXT`.
- **Hierarchy aware**: Every Segment knows its `parent_id`, `level` (depth), and `path` (breadcrumbs like ["Chapter 1", "Section 1.2"]).
- **Metadata rich**: Segments carry extracted metadata (titles, keywords) and origin info.

---

## 2. Installation

### 2.1. Prerequisites

- **Python**: Version 3.9 or higher.
- **System Tools**:
  - **Poppler**: Required for PDF parsing.
    - _Windows_: Download binary, add `bin/` to PATH.
    - _Linux_: `sudo apt-get install poppler-utils`
    - _Mac_: `brew install poppler`
  - **Ghostscript**: Required for `camelot-py` (Table extraction).
    - _Windows_: Install from official site.
    - _Linux_: `sudo apt-get install ghostscript`

### 2.2. Install from Source

Cloning the repository and installing in editable mode is recommended for development.

```bash
git clone https://github.com/your-org/rag-lib.git
cd rag-lib
pip install -e .
```

### 2.3. Install via Pip (Wheel)

If building a wheel:

```bash
pip install dist/rag_lib-0.1.0-py3-none-any.whl
```

### 2.4. Verify Installation

```bash
python -c "import rag_lib; print(rag_lib.__version__)"
```

---

## 3. Architecture

```mermaid
graph TD
    Input[Documents: PDF, DOCX, CSV] --> Loaders
    Loaders -->|Documents| Splitters
    Splitters -->|Segments| Chunkers
    Chunkers -->|Refined Segments| Processors
    Processors -->|Enriched Segments| Indexer
    Indexer --> VectorStore[(Vector DB)]

    subgraph "Core Components"
    Loaders[Loaders: PDF, Structured, CSV]
    Chunkers[Chunkers: Semantic, Regex, Markdown]
    Processors[Enricher: LLM Metadata]
    Indexer[Indexer: Parent-Child Retrieval]
    end
```

---

## 4. Core Components

### 4.1. Loaders (`rag_lib.loaders`)

Responsible for ingesting raw files and producing standard **LangChain Documents**.

| Class                      | Import Path                    | Description                                                                                                  |
| :------------------------- | :----------------------------- | :----------------------------------------------------------------------------------------------------------- |
| **`PDFLoader`**            | `rag_lib.loaders.pdf`          | Uses **Camelot** (lattice/stream) or **Poppler** to extract text and high-fidelity tables from PDFs.         |
| **`StructuredLoader`**     | `rag_lib.loaders.structured`   | Parses **DOCX** files into a single **Markdown** document, preserving hierarchy.                             |
| **`CSVLoader`**            | `rag_lib.loaders.csv_excel`    | Loads **CSV** files. Detects delimiters automatically and converts rows to Markdown tables.                  |
| **`ExcelLoader`**          | `rag_lib.loaders.csv_excel`    | Loads **Excel** files, treating each sheet as a segment.                                                     |
| **`RegexHierarchyLoader`** | `rag_lib.loaders.regex`        | Splits plain text files using a recursive list of regex patterns (e.g., Log files, Configs).                 |
| **`TableLoader`**          | `rag_lib.loaders.data_loaders` | Generic loader for structured tabular data (CSV/JSON). Supports "Row" mode (key-value text) or "Group" mode. |
| **`JsonLoader`**           | `rag_lib.loaders.data_loaders` | Loads entire JSON file content into a single **Document**.                                                   |
| **`QALoader`**             | `rag_lib.loaders.data_loaders` | Parses "Q: ... A: ..." formatted text files into QA segments.                                                |

### 4.2. Splitters & Chunkers (`rag_lib.chunkers`)

Refine raw Documents into Segments (Structure) or Segments into Chunks (Size).

- **`split_documents(docs)`**: Converts Documents -> Segments (Logical Splitting).
- **`split_segments(segs)`**: Converts Segments -> Chunks (Token/Character limitation).

| Class                                | Import Path                        | Description                                                                                |
| :----------------------------------- | :--------------------------------- | :----------------------------------------------------------------------------------------- |
| **`JsonSplitter`**                   | `rag_lib.chunkers.json`            | Splits JSON content based on **jq-style** schema (e.g., extracting items from a list).     |
| **`QASplitter`**                     | `rag_lib.chunkers.qa`              | Splits text containing "Q: ... A: ..." pairs into distinct Question/Answer segments.       |
| **`RegexHierarchySplitter`**         | `rag_lib.chunkers.regex_hierarchy` | Splits text recursively using a list of regex patterns (e.g. Log levels, Config sections). |
| **`SemanticChunker`**                | `rag_lib.chunkers.semantic`        | Splits text based on **cosine similarity** of sentence embeddings (detects topic shifts).  |
| **`RecursiveCharacterTextSplitter`** | `rag_lib.chunkers.recursive`       | Standard recursive splitting (paragraphs -> sentences -> words) to fit chunk size.         |
| **`TokenTextSplitter`**              | `rag_lib.chunkers.token`           | Splits text based on **token count** (using `tiktoken`) rather than characters.            |
| **`SentenceSplitter`**               | `rag_lib.chunkers.sentence`        | Uses **NLTK** to split by true sentences, grouping them until chunk size is reached.       |
| **`RegexSplitter`**                  | `rag_lib.chunkers.regex`           | Splits text based on a provided regex pattern.                                             |
| **`MarkdownTableSplitter`**          | `rag_lib.chunkers.markdown_table`  | Extracts Markdown tables from text to ensure they remain intact as distinct segments.      |

### 4.3. Processors (`rag_lib.processors`)

Enhance segments with additional intelligence.

| Class                 | Import Path                   | Description                                                                          |
| :-------------------- | :---------------------------- | :----------------------------------------------------------------------------------- |
| **`SegmentEnricher`** | `rag_lib.processors.enricher` | Uses an LLM to generate **Title**, **Keywords**, and a **Summary** for each segment. |

### 4.4. Indexer (`rag_lib.core.indexer`)

The bridge to your Vector Database.

- **`Indexer`**: Implements **Parent-Child Indexing**.
  - If a segment has a `summary` (from Enricher), it embeds the summary (Semantic Key) but stores the full content in the payload.
  - Supports both `index()` (sync) and `aindex()` (async).

---

## 5. Retrieval Components (`rag_lib.retrieval`)

`rag-lib` provides advanced retrieval strategies beyond simple similarity search.

### 5.1. Retrievers

| Class/Factory                    | Import Path                          | Description                                                                                                        |
| :------------------------------- | :----------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **`get_vector_retriever`**       | `rag_lib.retrieval.retrievers`       | Standard dense vector retrieval (Similarity, MMR, Score Threshold).                                                |
| **`get_bm25_retriever`**         | `rag_lib.retrieval.retrievers`       | **BM25** (Sparse) retrieval for keyword matching. (In-memory).                                                     |
| **`RegexRetriever`**             | `rag_lib.retrieval.retrievers`       | Finds documents matching a specific **Regex Pattern** (good for IDs/Codes).                                        |
| **`FuzzyRetriever`**             | `rag_lib.retrieval.retrievers`       | Uses **Levenshtein Distance** (RapidFuzz) for fuzzy string matching.                                               |
| **`ScoredMultiVectorRetriever`** | `rag_lib.retrieval.scored_retriever` | Multi-Vector Retriever that aggregates similarity scores (MAX) from chunks to parents. Supports `score_threshold`. |

### 5.2. Composition & Reranking

Located in `rag_lib.retrieval.composition`.

| Helper Function                            | Description                                                                                                                                       |
| :----------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`create_ensemble_retriever`**            | Combines multiple retrievers (e.g., BM25 + Vector) using **Reciprocal Rank Fusion (RRF)**.                                                        |
| **`create_reranking_retriever`**           | Wraps a retriever with a **Cross-Encoder** (e.g., BGE-Reranker) to re-score and filter the top-N results.                                         |
| **`create_dual_storage_retriever`**        | Sets up a **Multi-Vector** system: Searches low-dim vectors/summaries, but retrieves full documents from a separate Store (e.g., Redis/Postgres). |
| **`create_scored_dual_storage_retriever`** | Same as above, but returns `ScoredMultiVectorRetriever` which injects aggregated relevance scores into parent metadata.                           |

---

## 6. Usage Examples

### 6.1. Ingesting a Complex PDF with Tables

```python
from rag_lib.loaders.pdf import PDFLoader
from rag_lib.summarizers.table_llm import LLMTableSummarizer

# 1. Setup Table Summarizer (Optional)
summarizer = LLMTableSummarizer(llm=my_langchain_llm)

# 2. Load PDF
    summarizer=summarizer
)

docs = loader.load()

# 3. Create Segments
# Since each doc is a table (in table mode) or text, we can wrap them into Segments
# or split them further.
from rag_lib.chunkers.token import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=1000)
segments = splitter.split_documents(docs)

# resulting 'segments' list contains typed Segments.
```

### 6.2. Semantic Chunking & Indexing

```python
from rag_lib.chunkers.semantic import SemanticChunker
from rag_lib.processors.enricher import SegmentEnricher
from rag_lib.core.indexer import Indexer

# 1. Chunk Text Semantically
chunker = SemanticChunker(embeddings=my_embeddings, threshold=0.75)
text_segments = chunker.create_segments(raw_text) # Returns list of Segments

# 2. Enrich Segments (Generate Titles/Keywords)
enricher = SegmentEnricher(llm=my_chat_model)
idx = Indexer(vector_store=my_vector_store, embeddings=my_embeddings, enricher=enricher)

# 3. Index (Enrichment happens automatically inside indexer)
idx.index(text_segments, batch_size=50)
```

### 6.3. Hierarchical DOCX Parsing

```python
from rag_lib.loaders.structured import StructuredLoader
from rag_lib.chunkers.markdown_hierarchy import MarkdownHierarchySplitter

# Define Regex Patterns for custom sections (optional)
patterns = [
    {"level": 2, "pattern": r"^Article \d+"}, # Treat "Article X" as Level 2 functionality
]

loader = StructuredLoader("contract.docx") # Regex patterns now applied in Splitter if using RegexHierarchySplitter
docs = loader.load()

# Split by Hierarchy
# Use MarkdownHierarchySplitter to parse the headers from the loaded Markdown
splitter = MarkdownHierarchySplitter()
segments = splitter.split_documents(docs)

# Inspect hierarchy
for seg in segments:
    print(f"[{seg.level}] {seg.path} -> {seg.content[:30]}...")
```

---

## 7. Configuration & Dependencies

`rag-lib` relies on `rag_lib.config.Settings` (Pydantic) for global defaults:

- `ingestion.chunk_size`: Default CSV chunk size.
- `ingestion.semantic_threshold`: Default similarity threshold.
- `logging.level`: Global log level (INFO/DEBUG).

**Key Dependencies**:

- `camelot-py[cv]`: PDF Table extraction.
- `langchain-core`: Abstractions for LLMs and Embeddings.
- `pandas`: DataFrame operations.
- `nltk`: Sentence tokenization.

## 8. Extending the Library

To add a new Loader:

1.  Create `src/rag_lib/loaders/my_loader.py`.
2.  Implement a class with a `load(self) -> List[Segment]` method.
3.  Ensure you generate standard `Segment` objects with `uuid` and `type`.

To add a new Chunker:

1.  Inherit from `rag_lib.chunkers.base.TextSplitter` (if text-based).
2.  Implement `split_text(self, text: str) -> List[str]`.

---

## 9. Graph RAG Extensions

`rag-lib` 0.2.0+ introduces **Graph RAG** capabilities, inspired by _LightRAG_. This allows for entity-centric retrieval and global summarization.

### 9.1. Components (`rag_lib.graph`)

| Component                     | Description                                                            |
| :---------------------------- | :--------------------------------------------------------------------- |
| **`GraphNode` / `GraphEdge`** | Domain models for graph entities and relationships.                    |
| **`NetworkXGraphStore`**      | In-memory graph storage (default). Good for prototyping.               |
| **`Neo4jGraphStore`**         | Persistent graph storage adapter for Neo4j. Requires `rag-lib[graph]`. |

### 9.2. Processors

| Component                 | Description                                                                                                                               |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **`EntityExtractor`**     | LLM-based processor that extracts Entities and Relations from Segments and populates the GraphStore. Supports **Async Batch Processing**. |
| **`CommunityDetector`**   | Identifies communities (clusters) of nodes in the graph using modularity algorithms.                                                      |
| **`CommunitySummarizer`** | Generates high-level summaries for each detected community.                                                                               |

### 9.3. Retrieval

**`GraphRetriever`** supports two modes:

1.  **Local Mode** (`mode="local"`):
    - Implementation: Searches for specific entities (keywords) and traverses 1-hop or 2-hop neighbors.
    - Use Case: Specific questions about an entity (e.g., "Who follows user X?").

2.  **Global Mode** (`mode="global"`):
    - Implementation: Retrieves **Community Summaries** from the Vector Store.
    - Use Case: High-level thematic questions (e.g., "What are the main topics discussed?").

### 9.4. Example Usage

```python
from rag_lib.graph.store import NetworkXGraphStore
from rag_lib.processors.entity_extractor import EntityExtractor
from rag_lib.retrieval.graph_retriever import GraphRetriever

# 1. Setup
store = NetworkXGraphStore()
extractor = EntityExtractor(llm=my_llm, store=store)

# 2. Extract Graph
await extractor.aprocess_segments(segments)

# 3. Retrieve
retriever = GraphRetriever(store=store, mode="local")
docs = await retriever.ainvoke("Key Concept")
```

---

## 10. Advanced Loaders

### 10.1. MinerU (`rag_lib.loaders.miner_u`)

Integration with **Magic-PDF (MinerU)** for high-fidelity PDF parsing.

- **Purpose**: Extracts text, tables, and images while preserving layout information better than standard parsers.
- **Requirement**: `pip install rag-lib[miner_u]`
- **Class**: `MinerULoader`

```python
from rag_lib.loaders.miner_u import MinerULoader

loader = MinerULoader("complex_layout.pdf")
segments = loader.load()
# Returns typed Segments (TEXT, TABLE, IMAGE)
```

---

## 11. RAPTOR (Recursive Abstractive Processing)

**RAPTOR** (`rag_lib.raptor`) is a technique for **hierarchical summarization**. It recursively clusters text segments and summarizes them to build a "tree" of information. This allows the system to answer high-level questions that span across many documents.

### 11.1. Installation & Environment (Crucial!)

RAPTOR relies on `umap-learn` for dimensionality reduction, which has complex dependencies (Numba, llvmlite).

> [!WARNING]
> **Avoid `uv sync` for RAPTOR**: `uv` often fails to build `numba` because it requires specific C++ compilers.

**Recommended Installation**:

1.  **Use `pip` explicitly** inside your virtual environment to get pre-built wheels:

    ```bash
    # Windows / Linux / Mac
    pip install umap-learn scikit-learn
    ```

2.  **Project Config**:
    To install via project extras (if `uv` supports wheels on your OS):

    ```bash
    pip install ".[raptor]"
    # OR
    uv sync --extra raptor
    ```

3.  **Troubleshooting**:
    If you see `ImportError: No module named 'umap'`, run the manual `pip install` command above.

### 11.2. Components

| Component               | Import Path                    | Description                                                                  |
| :---------------------- | :----------------------------- | :--------------------------------------------------------------------------- |
| **`ClusteringService`** | `rag_lib.raptor.clustering`    | Uses **UMAP** (dim reduction) + **GMM** (Gaussian Mixture) to cluster texts. |
| **`ClusterSummarizer`** | `rag_lib.raptor.summarization` | Summarizes a list of texts using an LLM.                                     |
| **`TreeBuilder`**       | `rag_lib.raptor.tree_builder`  | Orchestrates the recursive Level 0 -> Level N tree creation.                 |
| **`RaptorProcessor`**   | `rag_lib.processors.raptor`    | Main entry point. Wraps `TreeBuilder` into a standard Processor.             |

### 11.3. Example Usage

```python
from rag_lib.processors.raptor import RaptorProcessor
from rag_lib.raptor.clustering import ClusteringService
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Setup Components
llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings()

# 2. Initialize Processor (Automatic Clustering Service)
raptor = RaptorProcessor(
    llm=llm,
    embeddings=embeddings,
    max_levels=3
)

# 3. Process Segments
# Input: List[Segment] (Level 0)
# Output: List[Segment] (Level 0 + Level 1 Summaries + Level 2 Summaries...)
enriched_segments = await raptor.aprocess(raw_segments)
```
