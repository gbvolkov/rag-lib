from typing import List, Tuple, Optional, Union, Dict
from rag_lib.core.domain import Document
from rag_lib.core.logger import logger

class RegexHierarchyLoader:
    """
    Refactored to return Document. Splitting logic moved to RegexHierarchySplitter.
    This loader now acts as a TextLoader, but accepts pattern args for backward compatibility (ignored).
    """
    def __init__(
        self, 
        file_path: str, 
        patterns: Union[List[Tuple[int, str]], List[Dict]], 
        exclude_patterns: Optional[List[str]] = None,
        include_parent_content: Union[bool, int] = False
    ):
        self.file_path = file_path
        # Patterns are effectively ignored here or stored for reference
        self.patterns = patterns 
        logger.warning("RegexHierarchyLoader now returns raw Document. Use RegexHierarchySplitter for splitting logic.")

    def load(self) -> List[Document]:
        """
        Reads file and returns list of Documents (1 per file).
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
             raise
             
        metadata = {
            "source": self.file_path,
            "patterns_config": str(self.patterns) # Storing config in metadata for reference?
        }
        return [Document(page_content=text, metadata=metadata)]

    def load_str(self, text: str) -> List[Document]:
        """
        Deprecated?
        """
        return [Document(page_content=text, metadata={"source": "string"})]
