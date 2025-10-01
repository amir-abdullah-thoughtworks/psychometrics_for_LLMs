# pip install sentence-transformers scikit-learn vendi-score numpy
import json
import re
import numpy as np
import zlib

from collections import Counter
from typing import List, Optional, Literal, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vendi_score import vendi


class DiversityMetrics:
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

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def ttr(self) -> float:
        tokens = [tok for t in self.texts for tok in self._tokenize(t)]
        if not tokens:
            return float("nan")
        return len(set(tokens)) / len(tokens)

    def distinct_n(self, n: int = 2) -> float:
        ngrams = []
        for t in self.texts:
            toks = self._tokenize(t)
            ngrams.extend(zip(*[toks[i:] for i in range(n)]))
        total = len(ngrams)
        return len(set(ngrams)) / total if total > 0 else float("nan")

    def mtld(self, threshold: float = 0.72) -> float:
        """Simplified MTLD implementation"""
        def mtld_calc(tokens):
            factors, start, types = 0, 0, set()
            for i, tok in enumerate(tokens, 1):
                types.add(tok)
                ttr_val = len(types) / i
                if ttr_val < threshold:
                    factors += 1
                    start, types = i, set()
            excess = len(tokens) - start
            return (len(tokens) - excess) / (factors + (excess / max(1, len(types))))
        tokens = [tok for t in self.texts for tok in self._tokenize(t)]
        if not tokens:
            return float("nan")
        return (mtld_calc(tokens) + mtld_calc(tokens[::-1])) / 2

    def yule_k(self) -> float:
        tokens = [tok for t in self.texts for tok in self._tokenize(t)]
        N = len(tokens)
        if N == 0:
            return float("nan")
        freqs = Counter(tokens)
        M1 = N
        M2 = sum(f * f for f in freqs.values())
        return 1e4 * (M2 - M1) / (M1 * M1)

    def compression_ratio(self) -> float:
        text_concat = " ".join(self.texts).encode("utf-8")
        if len(text_concat) == 0:
            return float("nan")
        comp = zlib.compress(text_concat)
        return len(comp) / len(text_concat)

    def avg_cosine_distance(self) -> float:
        """
        Average pairwise cosine distance across all embeddings.
        For normalized embeddings, cosine similarity = X @ X.T
        Distance = 1 - similarity.
        """
        X = self.embeddings
        n = len(X)
        if n < 2:
            return float("nan")

        # Cosine similarity matrix
        K = X @ X.T
        # Only upper triangle (exclude diagonal)
        i, j = np.triu_indices(n, k=1)
        sims = K[i, j]
        dists = 1.0 - sims
        return float(np.mean(dists))

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
    def vendi(self, kernel: str = "rbf", rbf_sigma: Optional[float] = None) -> float:
        X = self.embeddings
        n = len(X)
        if n == 0:
            return float("nan")

        if kernel == "cosine":
            # L2-normalized X -> cosine sim in [-1,1]; shift to [0,1] for nonnegativity
            S = (X @ X.T + 1.0) / 2.0
        elif kernel == "rbf":
            sq = np.sum(X ** 2, axis=1, keepdims=True)
            D2 = sq + sq.T - 2 * (X @ X.T)
            if rbf_sigma is None:
                tri = D2[np.triu_indices(n, k=1)]
                med = np.median(tri) if tri.size else 1.0
                rbf_sigma = np.sqrt(med / 2.0) if med > 1e-12 else 1.0
            S = np.exp(-D2 / (2.0 * (rbf_sigma ** 2)))
        else:
            raise ValueError("kernel must be 'cosine' or 'rbf'")

        # Symmetrize numerically
        K = 0.5 * (S + S.T)
        return float(vendi.score_K(K))

    # --- run all ---
    def compute_all(self, k_for_silhouette: int = 10, dump_file=False) -> Dict[str, Any]:
        scores = {
            "silhouette": self.silhouette(k=k_for_silhouette),
            "dcs_score": self.dcs(tau=0.07, kernel="cosine"),
            "vendi_score": self.vendi(),
            "ttr": self.ttr(),
            "compression_ratio": self.compression_ratio(),
            "yule_k": self.yule_k(),
            "mtld": self.mtld(),
            "distinct_n": self.distinct_n(),
            "avg_cosine_distance": self.avg_cosine_distance()
        }

        print(scores)

        if dump_file:
            with open("gradio_demos/personas_viewer/scores.json", "w") as f_out:
                json.dump(scores, f_out, indent=2)


# ---------------- Example ----------------
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast dark-colored fox leaps above a sleeping canine.",
        "Transformers are powerful models for language tasks.",
        "Neural networks can learn complex representations of text."
    ]
    tdm = DiversityMetrics(texts)
    scores = tdm.compute_all(k_for_silhouette=3)
    print(scores)
