# rag-lib

**rag-lib** is a specialized Python library for **Advanced RAG (Retrieval-Augmented Generation)** pipelines, focusing on hierarchical segmentation and layout-aware parsing.

## Installation

### Prerequisites

- Python 3.9+
- **Poppler** (for PDF parsing)
- **Ghostscript** (for table extraction)

### Installing from Source (Recommended)

To use `rag-lib` in other projects, you can install it directly from the repository using pip's editable mode or git support.

**Option 1: Editable Install (for development)**
Clone the repository and install:

```bash
git clone https://github.com/your-org/rag-lib.git
cd rag-lib
pip install -e .
```

**Option 2: Direct Install via Git**
Add to your `requirements.txt` or install command:

```bash
pip install git+https://github.com/your-org/rag-lib.git
```

## Usage

### Basic Example

```python
from rag_lib import PDFLoader, SemanticChunker, Indexer

# 1. Load Document
loader = PDFLoader(file_path="financial_report.pdf", backend="lattice")
segments = loader.load()

# 2. Chunk Text
chunker = SemanticChunker(embeddings=my_embeddings)
text_segments = chunker.split_text(raw_text)

# 3. Index
indexer = Indexer(vector_store=my_vector, embeddings=my_embeddings)
indexer.index(text_segments)
```

For more detailed documentation, please refer to [DEVELOPERS_GUIDE.md](DEVELOPERS_GUIDE.md).
