from enum import Enum
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import override

from langchain_classic.storage._lc_store import create_kv_docstore


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    similarity_score_threshold = "similarity_score_threshold"
    """Similarity search with a score threshold."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class ScoredMultiVectorRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""

    model_config = ConfigDict(extra="forbid")

    vectorstore: VectorStore
    """The underlying `VectorStore` to use to store small chunks
    and their embedding vectors"""

    byte_store: ByteStore | None = None
    """The lower-level backing storage layer for the parent documents"""

    docstore: BaseStore[str, Document]
    """The storage interface for the parent documents"""

    id_key: str = "doc_id"

    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""

    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""

    score_threshold: float | None = None
    """Threshold for similarity search"""

    @model_validator(mode="before")
    @classmethod
    def _shim_docstore(cls, values: dict) -> Any:
        if "search_threshold" in values:
            msg = "`search_threshold` is not supported. Use `score_threshold`."
            raise ValueError(msg)

        byte_store = values.get("byte_store")
        docstore = values.get("docstore")
        if byte_store is not None:
            docstore = create_kv_docstore(byte_store)
        elif docstore is None:
            msg = "You must pass a `byte_store` parameter."
            raise ValueError(msg)
        values["docstore"] = docstore
        return values

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents.
        """

        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            kwargs = self.search_kwargs.copy()
            if self.score_threshold is not None and "score_threshold" not in kwargs:
                kwargs["score_threshold"] = self.score_threshold

            sub_docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    **kwargs,
                )
            )
            seen: dict[str, list[Document]] = {}
            parent_max: dict[str, float] = {}

            for d, s in sub_docs_and_similarities:
                d.metadata["similarity_score"] = s
                if (pid := d.metadata.get(self.id_key)) is None:
                    continue
                pid = str(pid)    

                seen.setdefault(pid, []).append(d)

                cur = parent_max.get(pid)
                if cur is None or s > cur:
                    parent_max[pid] = s
                    for doc in seen[pid]:
                        doc.metadata["max_similarity_score"] = s
                else:
                    d.metadata["max_similarity_score"] = cur

            sub_docs = [d for d, _ in sub_docs_and_similarities]            
        else:
            sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the IDs that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        result: list[Document] = []
        for pid, doc in zip(ids, docs):
            if doc is None:
                continue
            if self.search_type == SearchType.similarity_score_threshold:
                # store max score of underlying chunks on the parent doc
                doc.metadata["max_similarity_score"] = parent_max.get(pid)
            result.append(doc)
        return result

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents.
        """
        if self.search_type == SearchType.mmr:
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query,
                **self.search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            kwargs = self.search_kwargs.copy()
            if self.score_threshold is not None and "score_threshold" not in kwargs:
                kwargs["score_threshold"] = self.score_threshold

            sub_docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query,
                    **kwargs,
                )
            )
            seen: dict[str, list[Document]] = {}
            parent_max: dict[str, float] = {}

            for d, s in sub_docs_and_similarities:
                d.metadata["similarity_score"] = s
                if (pid := d.metadata.get(self.id_key)) is None:
                    continue
                pid = str(pid)

                seen.setdefault(pid, []).append(d)

                cur = parent_max.get(pid)
                if cur is None or s > cur:
                    parent_max[pid] = s
                    for doc in seen[pid]:
                        doc.metadata["max_similarity_score"] = s
                else:
                    d.metadata["max_similarity_score"] = cur

            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query,
                **self.search_kwargs,
            )

        # We do this to maintain the order of the IDs that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        result: list[Document] = []
        for pid, doc in zip(ids, docs):
            if doc is None:
                continue
            if self.search_type == SearchType.similarity_score_threshold:
                # store max score of underlying chunks on the parent doc
                doc.metadata["max_similarity_score"] = parent_max.get(pid)
            result.append(doc)
        return result
