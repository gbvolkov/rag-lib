__all__ = ["CSVTableSplitter", "HTMLSplitter"]


def __getattr__(name: str):
    if name == "CSVTableSplitter":
        from rag_lib.chunkers.csv_table import CSVTableSplitter

        return CSVTableSplitter
    if name == "HTMLSplitter":
        from rag_lib.chunkers.html import HTMLSplitter

        return HTMLSplitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
