# pip install sentence-transformers scikit-learn vendi-score numpy

import numpy as np
from typing import List, Optional, Literal, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vendi_score import vendi_score


class TextDiversityMetricsST:
    """
    Text diversity metrics using Sentence-Transformers embeddings (Qwen/Qwen3-Embedding-0.6B).
    Provides:
      - silhouette(k): cosine silhouette coefficient
      - dcs(tau, kernel): DCScore (Shaib et al. 2025)
      - vendi(): Vendi Score (Friedman & Dieng 2023)
    """

    def __init__(
        self,
        texts: List[str],
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.texts = texts
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = np.asarray(
            self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=normalize,
            )
        )

    # --- silhouette score ---
    def silhouette(self, k: int = 10) -> float:
        X = self.embeddings
        n = len(X)
        if n < 3 or k < 2:
            return float("nan")
        k = min(k, n - 1)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2:
            return float("nan")
        return float(silhouette_score(X, labels, metric="cosine"))

    # --- DCS score ---
    def dcs(
        self,
        tau: float = 0.07,
        kernel: Literal["cosine", "rbf"] = "cosine",
        rbf_sigma: Optional[float] = None,
    ) -> float:
        X = self.embeddings
        if len(X) == 0:
            return float("nan")

        if kernel == "cosine":
            K = X @ X.T
        elif kernel == "rbf":
            sq = np.sum(X**2, axis=1, keepdims=True)
            D2 = sq + sq.T - 2 * (X @ X.T)
            if rbf_sigma is None:
                tri = D2[np.triu_indices(len(X), k=1)]
                med = np.median(tri) if tri.size else 1.0
                rbf_sigma = np.sqrt(max(med, 1e-12) / 2.0)
            K = np.exp(-D2 / (2.0 * rbf_sigma**2))
        else:
            raise ValueError("kernel must be 'cosine' or 'rbf'")

        Z = K / max(tau, 1e-12)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        return float(np.trace(P))

    # --- Vendi score ---
    def vendi(self) -> float:
        return float(vendi_score(self.embeddings))

    # --- run all ---
    def compute_all(self, k_for_silhouette: int = 10) -> Dict[str, Any]:
        return {
            "silhouette": self.silhouette(k=k_for_silhouette),
            "dcs_score": self.dcs(tau=0.07, kernel="cosine"),
            "vendi_score": self.vendi(),
        }


# ---------------- Example ----------------
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast dark-colored fox leaps above a sleeping canine.",
        "Transformers are powerful models for language tasks.",
        "Neural networks can learn complex representations of text."
    ]
    tdm = TextDiversityMetricsST(texts)
    print(tdm.compute_all(k_for_silhouette=3))
