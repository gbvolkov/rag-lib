from typing import List, Optional, Union, Dict
from rag_lib.chunkers.regex_hierarchy import RegexHierarchySplitter

class MarkdownHierarchySplitter(RegexHierarchySplitter):
    """
    Splits Markdown text based on headers (#, ##, etc.) preserving hierarchy.
    Subclass of RegexHierarchySplitter with pre-configured patterns.
    """
    def __init__(
        self,
        exclude_code_blocks: bool = True,
        include_parent_content: Union[bool, int] = False
    ):
        """
        :param exclude_code_blocks: If True, attempts to avoid splitting inside code blocks (naive check).
        :param include_parent_content: Prefix parent content to children.
        """
        # Define patterns for headers level 1 to 6
        patterns = [
            (1, r'^#\s+(.+)'),
            (2, r'^##\s+(.+)'),
            (3, r'^###\s+(.+)'),
            (4, r'^####\s+(.+)'),
            (5, r'^#####\s+(.+)'),
            (6, r'^######\s+(.+)')
        ]
        
        exclude_patterns = []
        # Naive approach to ignore lines that look like headers but are inside code blocks?
        # Regex is line-based, so statefulness is hard.
        # But for now, we rely on the line-based regex of parent.
        
        super().__init__(
            patterns=patterns,
            exclude_patterns=exclude_patterns, # TODO: Robust Code Block handling requires state
            include_parent_content=include_parent_content
        )
