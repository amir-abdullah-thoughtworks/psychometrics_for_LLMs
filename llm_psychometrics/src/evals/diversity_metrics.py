# pip install torch transformers scikit-learn numpy scipy

from typing import List, Optional, Literal, Dict, Any
import math
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # mean pool with attention mask
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B, 1)
    return summed / counts


class DiversityMetrics:
    """
    Initialize with texts -> precompute embeddings using Qwen/Qwen3-Embedding-0.6B.

    Metrics:
      - silhouette(k=10): cosine silhouette over KMeans clusters
      - dcs(tau=0.07, kernel='cosine'|'rbf'): softmax-trace of similarity matrix
      - vendi(kernel='rbf'|'cosine'): exp(Shannon entropy of eigenvalues of normalized similarity)

    Notes:
      * We compute mean-pooled embeddings from the last_hidden_state and L2-normalize them.
      * For Vendi, an RBF kernel is safest (PSD). Cosine is shifted to [0,1] as (cos+1)/2.
    """

    def __init__(
        self,
        texts: List[str],
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        max_length: int = 512,
        fp16: bool = True,
        tqdm_bar: bool = True,
    ):
        self.texts = texts
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        # embed all texts
        self.embeddings = self._embed_all(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            fp16=fp16,
            tqdm_bar=tqdm_bar,
        )  # (N, D), L2-normalized if normalize=True

    @torch.no_grad()
    def _embed_all(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        fp16: bool,
        tqdm_bar: bool,
    ) -> np.ndarray:
        n = len(texts)
        out = []
        rng = range(0, n, batch_size)

        pbar = None
        if tqdm_bar:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(rng, desc="Embedding texts")
            except Exception:
                pbar = rng

        it = pbar if pbar is not None else rng

        for s in it:
            e = min(n, s + batch_size)
            batch = texts[s:e]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.autocast(device_type=self.device if self.device != "mps" else "cpu", dtype=torch.float16, enabled=fp16 and self.device != "cpu"):
                out_dict = self.model(**enc)
                if hasattr(out_dict, "last_hidden_state"):
                    hidden = out_dict.last_hidden_state      # (B, T, H)
                else:
                    # Some embedding models return tuple; fall back
                    hidden = out_dict[0]

                pooled = _mean_pool(hidden, enc["attention_mask"])  # (B, H)
                if self.normalize:
                    pooled = nn.functional.normalize(pooled, p=2, dim=1)

            out.append(pooled.detach().cpu().float().numpy())

        X = np.vstack(out) if out else np.zeros((0, 1024), dtype=np.float32)
        return X

    # ---------- Utilities for kernels ----------
    @staticmethod
    def _cosine_sim(X: np.ndarray) -> np.ndarray:
        # assumes rows are L2-normalized
        return X @ X.T

    @staticmethod
    def _rbf_sim(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        sq = np.sum(X**2, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (X @ X.T)
        if sigma is None:
            tri = D2[np.triu_indices(len(X), k=1)]
            med = np.median(tri) if tri.size else 1.0
            sigma = math.sqrt(max(med, 1e-12) / 2.0)
        return np.exp(-D2 / (2.0 * (sigma ** 2)))

    # ---------- Metrics ----------
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

    def dcs(
        self,
        tau: float = 0.07,
        kernel: Literal["cosine", "rbf"] = "cosine",
        rbf_sigma: Optional[float] = None,
    ) -> float:
        X = self.embeddings
        n = len(X)
        if n == 0:
            return float("nan")
        if kernel == "cosine":
            K = self._cosine_sim(X)
        elif kernel == "rbf":
            K = self._rbf_sim(X, sigma=rbf_sigma)
        else:
            raise ValueError("kernel must be 'cosine' or 'rbf'")

        Z = K / max(tau, 1e-12)
        Z = Z - Z.max(axis=1, keepdims=True)  # numerical stability
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        return float(np.trace(P))

    def vendi(
        self,
        kernel: Literal["rbf", "cosine"] = "rbf",
        rbf_sigma: Optional[float] = None,
    ) -> float:
        X = self.embeddings
        n = len(X)
        if n == 0:
            return float("nan")

        if kernel == "rbf":
            S = self._rbf_sim(X, sigma=rbf_sigma)
        elif kernel == "cosine":
            # map cosine [-1,1] -> [0,1] to keep nonnegativity; symmetrize
            S = (self._cosine_sim(X) + 1.0) / 2.0
        else:
            raise ValueError("kernel must be 'rbf' or 'cosine'")

        K = 0.5 * (S + S.T)
        evals = np.linalg.eigvalsh(K)
        evals = np.clip(evals, 0.0, None)
        s = evals.sum()
        if s <= 1e-12:
            return 0.0
        p = evals / s
        mask = p > 0
        H = -float(np.sum(p[mask] * np.log(p[mask])))  # natural log
        return float(np.exp(H))

    def compute_all(self, k_for_silhouette: int = 10) -> Dict[str, Any]:
        return {
            "silhouette": self.silhouette(k=k_for_silhouette),
            "dcs_score": self.dcs(tau=0.07, kernel="cosine"),
            "vendi_score": self.vendi(kernel="rbf"),
        }


# ---------------- Example ----------------
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast dark-colored fox leaps above a sleeping canine.",
        "Transformers are powerful models for language tasks.",
        "Neural networks can learn complex representations of text.",
        "This sentence is quite different from the others."
    ]
    tdm = DiversityMetrics(texts)  # embeds on init with Qwen/Qwen3-Embedding-0.6B
    print(tdm.compute_all(k_for_silhouette=3))
