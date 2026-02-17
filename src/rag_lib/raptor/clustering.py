import numpy as np
from typing import Optional, List, Tuple

import umap
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 224

class ClusteringService:
    """
    Service for clustering embeddings using UMAP (dimensionality reduction) 
    and Gaussian Mixture Models (soft clustering).
    """

    def __init__(self):
        # RAPTOR dependencies are required; missing deps should fail at import time.
        pass

    def perform_clustering(
        self, 
        embeddings: np.ndarray, 
        dim: int, 
        threshold: float
    ) -> List[np.ndarray]:
        """
        Perform recursive clustering (Global -> Local).
        Returns a list of cluster IDs for each embedding.
        """
        if len(embeddings) <= dim + 1:
            # Not enough data for UMAP/GMM
            return [np.array([0]) for _ in range(len(embeddings))]

        # 1. Global Cluster
        reduced_global = self._global_reduction(embeddings, dim)
        global_clusters, n_global = self._gmm_cluster(reduced_global, threshold)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # 2. Local Cluster within each Global Cluster
        for i in range(n_global):
            # Mask for current global cluster
            mask = np.array([i in gc for gc in global_clusters])
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) == 0:
                continue
            
            if len(cluster_embeddings) <= dim + 1:
                local_clusters = [np.array([0]) for _ in cluster_embeddings]
                n_local = 1
            else:
                reduced_local = self._local_reduction(cluster_embeddings, dim)
                local_clusters, n_local = self._gmm_cluster(reduced_local, threshold)

            # Assign IDs back to original indices
            # Indices in original 'embeddings' array that belong to this global cluster
            global_indices = np.where(mask)[0]
            
            for j in range(n_local):
                # Mask for local cluster j within this global cluster
                local_mask = np.array([j in lc for lc in local_clusters])
                
                # Get the indices relative to cluster_embeddings
                local_relative_indices = np.where(local_mask)[0]
                
                # Map back to global indices
                final_indices = global_indices[local_relative_indices]
                
                for idx in final_indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local

        return all_local_clusters

    def _global_reduction(
        self, 
        embeddings: np.ndarray, 
        dim: int, 
        n_neighbors: Optional[int] = None, 
        metric: str = "cosine"
    ) -> np.ndarray:
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        
        # UMAP safety check
        n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=dim, 
            metric=metric, 
            random_state=RANDOM_SEED
        )
        return reducer.fit_transform(embeddings)

    def _local_reduction(
        self, 
        embeddings: np.ndarray, 
        dim: int, 
        num_neighbors: int = 10, 
        metric: str = "cosine"
    ) -> np.ndarray:
        # UMAP safety check
        num_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
        
        reducer = umap.UMAP(
            n_neighbors=num_neighbors, 
            n_components=dim, 
            metric=metric, 
            random_state=RANDOM_SEED
        )
        return reducer.fit_transform(embeddings)

    def _get_optimal_clusters(
        self, 
        embeddings: np.ndarray, 
        max_clusters: int = 50
    ) -> int:
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=RANDOM_SEED)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return int(n_clusters[np.argmin(bics)])

    def _gmm_cluster(
        self, 
        embeddings: np.ndarray, 
        threshold: float
    ) -> Tuple[List[np.ndarray], int]:
        n_clusters = self._get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters
