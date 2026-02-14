from typing import List, Dict, Tuple, Optional
import numpy as np
from langchain_core.embeddings import Embeddings
from rag_lib.core.domain import Segment, SegmentType
from rag_lib.raptor.clustering import ClusteringService
from rag_lib.raptor.summarization import ClusterSummarizer
from rag_lib.core.logger import logger
import uuid

class TreeBuilder:
    """
    Builds a RAPTOR tree from a list of Segments.
    """
    def __init__(
        self, 
        clustering_service: ClusteringService, 
        summarizer: ClusterSummarizer,
        embeddings_model: Embeddings
    ):
        self.clustering = clustering_service
        self.summarizer = summarizer
        self.embeddings_model = embeddings_model

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
                summary_text = self.summarizer.summarize([s.content for s in cluster_segs])
                
                # Create Summary Segment
                summary_seg = Segment(
                    content=summary_text,
                    segment_id=str(uuid.uuid4()),
                    type=SegmentType.TEXT, # Or a specific SUMMARY type if we add it
                    metadata={
                        "raptor_level": level,
                        "raptor_cluster_id": int(cluster_id),
                        "raptor_child_ids": [s.segment_id for s in cluster_segs],
                        "is_raptor_summary": True
                    }
                )
                new_segments.append(summary_seg)
            
            # Add to results
            all_segments.extend(new_segments)
            
            # Prepare for next iteration
            current_level_segments = new_segments
            
            if len(new_segments) <= 1:
                logger.info("Reached top of tree (<= 1 summary node). Stopping.")
                break
                
        return all_segments

    async def abuild(self, segments: List[Segment], n_levels: int = 3) -> List[Segment]:
        """Async version of build"""
        # Similar logic but with async embedding and summarization
        if not segments:
            return []

        logger.info(f"Building RAPTOR tree (Async) with {len(segments)} leaves...")
        current_level_segments = segments
        all_segments = segments.copy()

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
                summary = await self.summarizer.asummarize([s.content for s in csegs])
                return Segment(
                    content=summary,
                    segment_id=str(uuid.uuid4()),
                    type=SegmentType.TEXT,
                    metadata={
                        "raptor_level": level,
                        "raptor_cluster_id": int(cid),
                        "raptor_child_ids": [s.segment_id for s in csegs],
                        "is_raptor_summary": True
                    }
                )

            tasks = [_process_cluster(cid, csegs) for cid, csegs in cluster_map.items()]
            if tasks:
                new_segments = await asyncio.gather(*tasks)
            
            all_segments.extend(new_segments)
            current_level_segments = list(new_segments)
            
            if len(new_segments) <= 1:
                break
        
        return all_segments

    def _embed_segments(self, segments: List[Segment]) -> np.ndarray:
        texts = [s.content for s in segments]
        embeddings = self.embeddings_model.embed_documents(texts)
        return np.array(embeddings)
