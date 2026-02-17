from typing import List, Dict
import numpy as np
from langchain_core.embeddings import Embeddings
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.raptor.clustering import ClusteringService
from rag_lib.raptor.summarization import ClusterSummarizer
from rag_lib.core.logger import logger
import uuid
from collections import deque
import re

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

class TreeBuilder:
    """
    Builds a RAPTOR tree from a list of Segments.
    """
    def __init__(
        self, 
        clustering_service: ClusteringService, 
        summarizer: ClusterSummarizer,
        embeddings_model: Embeddings,
        summary_target_ratio: float = 0.35,
        summary_max_chars: int = 1200,
        summary_min_chars: int = 120,
        summary_preserve_language: bool = True,
        strict_quality: bool = True,
    ):
        if not 0 < summary_target_ratio < 1:
            raise ValueError("summary_target_ratio must be in range (0, 1).")
        if summary_min_chars <= 0:
            raise ValueError("summary_min_chars must be > 0.")
        if summary_max_chars < summary_min_chars:
            raise ValueError("summary_max_chars must be >= summary_min_chars.")

        self.clustering = clustering_service
        self.summarizer = summarizer
        self.embeddings_model = embeddings_model
        self.summary_target_ratio = summary_target_ratio
        self.summary_max_chars = summary_max_chars
        self.summary_min_chars = summary_min_chars
        self.summary_preserve_language = summary_preserve_language
        self.strict_quality = strict_quality

    def build(self, segments: List[Segment], n_levels: int = 3) -> List[Segment]:
        """
        Builds the tree and returns the flat list of all segments (leaves + summaries).
        """
        if not segments:
            return []
            
        logger.info(f"Building RAPTOR tree with {len(segments)} leaf segments, max_levels={n_levels}")
        
        # Initialize with leaves at level 0
        current_level_segments = segments
        all_segments = segments.copy()
        
        document_language = self._detect_document_language(segments)

        for level in range(1, n_levels + 1):
            logger.info(f"Processing RAPTOR Level {level}...")
            
            # 1. Embed current level
            embeddings = self._embed_segments(current_level_segments)
            
            # 2. Cluster
            # Dim=10, Threshold=0.1 are defaults from paper/reference
            cluster_assignments = self.clustering.perform_clustering(
                embeddings, dim=10, threshold=0.1
            )
            
            # 3. Group Segments by Cluster
            n_clusters = len(set(c for sublist in cluster_assignments for c in sublist))
            logger.info(f"Generated {n_clusters} clusters at level {level}")
            
            if n_clusters <= 1 and len(current_level_segments) > 1:
                # If everything collapsed to 1 cluster, we might be done or at top
                # But if we have >1 segments and 1 cluster, we still summarize it once.
                pass
            elif n_clusters == 0:
                break

            # Map cluster_id -> List[Segment]
            cluster_map: Dict[int, List[Segment]] = {}
            for idx, clusters in enumerate(cluster_assignments):
                seg = current_level_segments[idx]
                for cluster_id in clusters:
                    if cluster_id not in cluster_map:
                        cluster_map[cluster_id] = []
                    cluster_map[cluster_id].append(seg)

            # 4. Summarize each cluster -> New Segments
            new_segments = []
            for cluster_id, cluster_segs in cluster_map.items():
                summary_seg = self._build_summary_segment(
                    level=level,
                    cluster_id=int(cluster_id),
                    cluster_segs=cluster_segs,
                    document_language=document_language,
                )
                new_segments.append(summary_seg)
            
            # Add to results
            all_segments.extend(new_segments)
            
            # Prepare for next iteration
            current_level_segments = new_segments
            
            if len(new_segments) <= 1:
                logger.info("Reached top of tree (<= 1 summary node). Stopping.")
                break
                
        self._finalize_tree_hierarchy(all_segments)
        return all_segments

    async def abuild(self, segments: List[Segment], n_levels: int = 3) -> List[Segment]:
        """Async version of build"""
        # Similar logic but with async embedding and summarization
        if not segments:
            return []

        logger.info(f"Building RAPTOR tree (Async) with {len(segments)} leaves...")
        current_level_segments = segments
        all_segments = segments.copy()
        document_language = self._detect_document_language(segments)

        for level in range(1, n_levels + 1):
            logger.info(f"Processing RAPTOR Level {level}...")
            
            # 1. Embed (Async if supported, else sync)
            if hasattr(self.embeddings_model, "aembed_documents"):
                texts = [s.content for s in current_level_segments]
                embeddings_list = await self.embeddings_model.aembed_documents(texts)
                embeddings = np.array(embeddings_list)
            else:
                embeddings = self._embed_segments(current_level_segments)

            # 2. Cluster (CPU bound, keep sync or run in executor if needed)
            cluster_assignments = self.clustering.perform_clustering(embeddings, dim=10, threshold=0.1)
            
            # 3. Group
            cluster_map: Dict[int, List[Segment]] = {}
            for idx, clusters in enumerate(cluster_assignments):
                seg = current_level_segments[idx]
                for cluster_id in clusters:
                    if cluster_id not in cluster_map:
                        cluster_map[cluster_id] = []
                    cluster_map[cluster_id].append(seg)
            
            # 4. Summarize (Async)
            # Parallelize summarization
            import asyncio
            new_segments = []
            
            async def _process_cluster(cid, csegs):
                return await self._abuild_summary_segment(
                    level=level,
                    cluster_id=int(cid),
                    cluster_segs=csegs,
                    document_language=document_language,
                )

            tasks = [_process_cluster(cid, csegs) for cid, csegs in cluster_map.items()]
            if tasks:
                new_segments = await asyncio.gather(*tasks)
            
            all_segments.extend(new_segments)
            current_level_segments = list(new_segments)
            
            if len(new_segments) <= 1:
                break
        
        self._finalize_tree_hierarchy(all_segments)
        return all_segments

    def _detect_document_language(self, segments: List[Segment]) -> str:
        if not self.summary_preserve_language:
            return "english"
        combined_text = "\n".join(seg.content for seg in segments if seg.content)
        return self._detect_primary_script_language(combined_text, default="english")

    def _detect_primary_script_language(self, text: str, default: str = "english") -> str:
        cyr = len(_CYRILLIC_RE.findall(text))
        lat = len(_LATIN_RE.findall(text))
        total = cyr + lat
        if total == 0:
            return default
        if cyr > lat and (cyr / total) >= 0.30:
            return "russian"
        if lat >= cyr and (lat / total) >= 0.30:
            return "english"
        return default

    def _compute_summary_budget(self, children_chars: int) -> int:
        ratio_budget = max(1, int(children_chars * self.summary_target_ratio))
        budget = min(self.summary_max_chars, ratio_budget)

        if children_chars > self.summary_min_chars:
            budget = max(budget, self.summary_min_chars)
        if children_chars > 1:
            budget = min(budget, children_chars - 1)
        return max(1, budget)

    def _validate_summary(
        self,
        *,
        summary_text: str,
        children_chars: int,
        max_chars: int,
        target_language: str,
        level: int,
        cluster_id: int,
    ) -> tuple[int, str, bool]:
        summary_chars = len(summary_text)
        detected_language = self._detect_primary_script_language(
            summary_text,
            default=target_language,
        )
        exceeds_max_chars = summary_chars > max_chars

        hard_violations: List[str] = []
        if summary_chars == 0:
            hard_violations.append("summary is empty")
        if children_chars > 0 and summary_chars >= children_chars:
            hard_violations.append(
                f"summary_chars={summary_chars} >= children_chars={children_chars}"
            )
        if self.summary_preserve_language and detected_language != target_language:
            hard_violations.append(
                f"summary_language={detected_language} != target_language={target_language}"
            )

        if exceeds_max_chars:
            logger.warning(
                "RAPTOR summary exceeded max_chars at level=%s cluster_id=%s "
                "(summary_chars=%s, max_chars=%s). Continuing.",
                level,
                cluster_id,
                summary_chars,
                max_chars,
            )

        if hard_violations:
            details = "; ".join(hard_violations)
            message = (
                f"RAPTOR summary validation failed at level={level}, "
                f"cluster_id={cluster_id}: {details}"
            )
            if self.strict_quality:
                raise ValueError(message)
            logger.warning(message)

        return summary_chars, detected_language, exceeds_max_chars

    def _build_summary_segment(
        self,
        *,
        level: int,
        cluster_id: int,
        cluster_segs: List[Segment],
        document_language: str,
    ) -> Segment:
        child_texts = [s.content for s in cluster_segs]
        children_chars = sum(len(text) for text in child_texts)
        max_chars = self._compute_summary_budget(children_chars)
        target_language = document_language if self.summary_preserve_language else "english"

        summary_text = self.summarizer.summarize(
            child_texts,
            target_language=target_language,
            max_chars=max_chars,
            target_ratio=self.summary_target_ratio,
        )
        summary_text = summary_text.strip()
        summary_chars, summary_language, exceeds_max_chars = self._validate_summary(
            summary_text=summary_text,
            children_chars=children_chars,
            max_chars=max_chars,
            target_language=target_language,
            level=level,
            cluster_id=cluster_id,
        )

        compression_ratio = (
            round(summary_chars / children_chars, 4) if children_chars else 0.0
        )
        return Segment(
            content=summary_text,
            segment_id=str(uuid.uuid4()),
            type=SegmentType.TEXT,
            metadata={
                "raptor_level": level,
                "raptor_cluster_id": cluster_id,
                "raptor_child_ids": [s.segment_id for s in cluster_segs],
                "is_raptor_summary": True,
                "raptor_summary_chars": summary_chars,
                "raptor_children_chars": children_chars,
                "raptor_compression_ratio": compression_ratio,
                "raptor_summary_language": summary_language,
                "raptor_summary_max_chars": max_chars,
                "raptor_summary_exceeds_max_chars": exceeds_max_chars,
            },
        )

    async def _abuild_summary_segment(
        self,
        *,
        level: int,
        cluster_id: int,
        cluster_segs: List[Segment],
        document_language: str,
    ) -> Segment:
        child_texts = [s.content for s in cluster_segs]
        children_chars = sum(len(text) for text in child_texts)
        max_chars = self._compute_summary_budget(children_chars)
        target_language = document_language if self.summary_preserve_language else "english"

        summary_text = await self.summarizer.asummarize(
            child_texts,
            target_language=target_language,
            max_chars=max_chars,
            target_ratio=self.summary_target_ratio,
        )
        summary_text = summary_text.strip()
        summary_chars, summary_language, exceeds_max_chars = self._validate_summary(
            summary_text=summary_text,
            children_chars=children_chars,
            max_chars=max_chars,
            target_language=target_language,
            level=level,
            cluster_id=cluster_id,
        )

        compression_ratio = (
            round(summary_chars / children_chars, 4) if children_chars else 0.0
        )
        return Segment(
            content=summary_text,
            segment_id=str(uuid.uuid4()),
            type=SegmentType.TEXT,
            metadata={
                "raptor_level": level,
                "raptor_cluster_id": cluster_id,
                "raptor_child_ids": [s.segment_id for s in cluster_segs],
                "is_raptor_summary": True,
                "raptor_summary_chars": summary_chars,
                "raptor_children_chars": children_chars,
                "raptor_compression_ratio": compression_ratio,
                "raptor_summary_language": summary_language,
                "raptor_summary_max_chars": max_chars,
                "raptor_summary_exceeds_max_chars": exceeds_max_chars,
            },
        )

    def _embed_segments(self, segments: List[Segment]) -> np.ndarray:
        texts = [s.content for s in segments]
        embeddings = self.embeddings_model.embed_documents(texts)
        return np.array(embeddings)

    def _finalize_tree_hierarchy(self, segments: List[Segment]) -> None:
        """
        Populate parent links and normalize Segment.level as top-down depth.

        Notes:
        - Keep `metadata["raptor_level"]` as-is (bottom-up RAPTOR iteration level).
        - Add `metadata["raptor_parent_ids"]` for all non-root nodes.
        - Set `Segment.parent_id` when there is exactly one parent.
        - Set `Segment.level` and `metadata["raptor_depth_from_root"]` where
          root summary nodes are depth 0 and leaves are deepest.
        """
        id_to_segment: Dict[str, Segment] = {
            seg.segment_id: seg for seg in segments if seg.segment_id
        }
        if not id_to_segment:
            return

        children_by_parent: Dict[str, List[str]] = {}
        parent_ids_by_child: Dict[str, List[str]] = {sid: [] for sid in id_to_segment}

        for seg in segments:
            seg_id = seg.segment_id
            if not seg_id:
                continue

            raw_child_ids = seg.metadata.get("raptor_child_ids", [])
            if isinstance(raw_child_ids, str):
                raw_child_ids = [c.strip() for c in raw_child_ids.split(",") if c.strip()]
            elif not isinstance(raw_child_ids, list):
                raw_child_ids = []

            valid_child_ids: List[str] = []
            for child_id in raw_child_ids:
                if child_id in id_to_segment:
                    valid_child_ids.append(child_id)
                    parent_ids_by_child[child_id].append(seg_id)

            if valid_child_ids:
                children_by_parent[seg_id] = valid_child_ids

            # Leaves do not have RAPTOR summary metadata by default.
            seg.metadata.setdefault("raptor_level", 0)

        # Write parent relationships.
        for seg_id, seg in id_to_segment.items():
            parent_ids = parent_ids_by_child.get(seg_id, [])
            if parent_ids:
                seg.metadata["raptor_parent_ids"] = parent_ids
                if len(parent_ids) == 1:
                    seg.parent_id = parent_ids[0]
                else:
                    # parent_id is singular; keep None when there are many.
                    seg.parent_id = None
            else:
                seg.metadata.pop("raptor_parent_ids", None)
                seg.parent_id = None

        # Compute top-down depth from root summaries.
        root_ids = [sid for sid, parents in parent_ids_by_child.items() if not parents]
        if not root_ids:
            # Defensive fallback for malformed graphs.
            root_ids = list(id_to_segment.keys())

        depth_by_id: Dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((rid, 0) for rid in root_ids)

        while queue:
            node_id, depth = queue.popleft()
            prev = depth_by_id.get(node_id)
            if prev is not None and prev <= depth:
                continue

            depth_by_id[node_id] = depth
            for child_id in children_by_parent.get(node_id, []):
                queue.append((child_id, depth + 1))

        for seg_id, seg in id_to_segment.items():
            depth = depth_by_id.get(seg_id, 0)
            seg.level = depth
            seg.metadata["raptor_depth_from_root"] = depth
