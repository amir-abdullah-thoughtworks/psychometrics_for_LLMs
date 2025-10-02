# Uses spaCy for lemmatized tokens (no punctuation), Sentence-Transformers (Qwen/Qwen3-Embedding-0.6B),
# and vendi-score's API. Keeps your method names & compute_all signature.

import json
import numpy as np
import zlib
from collections import Counter
from typing import List, Optional, Literal, Dict, Any

import spacy
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vendi_score import vendi


class DiversityMetrics:
    """
    Text diversity metrics using Sentence-Transformers embeddings (Qwen/Qwen3-Embedding-0.6B)
    and spaCy-based tokenization with lemmatization (punctuation stripped).

    Provides:
      - silhouette(k): cosine silhouette coefficient
      - dcs(tau, kernel): DCScore
      - vendi(): Vendi Score
      - per_text_ttr(), cumulative_ttr(), msttr(segment_size)
      - mtld(), yule_k(), distinct_n(n), compression_ratio()
      - avg_cosine_distance()
    """

    def __init__(
        self,
        texts: List[str],
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = "cuda",
        normalize: bool = True,
        spacy_model: str = "en_core_web_sm",
        spacy_disable: Optional[List[str]] = None,  # e.g., ["ner"]
        remove_stopwords: bool = False,
        batch_size: int = 32
    ):
        self.texts = texts
        self.remove_stopwords = remove_stopwords

        # --- embeddings once ---
        print(f"Loading onto {device}.")
        self.model = SentenceTransformer(model_name, device=device)
        print("Generating embeddings...")

        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i : i + batch_size]
            embs = self.model.encode(
                batch,
                show_progress_bar=False,  # disable internal tqdm
                normalize_embeddings=normalize, convert_to_tensor=True,
            ).cpu()
            all_embs.append(embs)

        self.embeddings = np.vstack(all_embs)



        print(f"Done encoding {len(self.texts)} texts into embeddings")

        # --- spaCy once (lemmatized tokens; no punctuation/spaces) ---
        spacy_disable = spacy_disable or []
        self.nlp = spacy.load(spacy_model, disable=spacy_disable)

        self.lemmas_per_doc: List[List[str]] = []
        for doc in self.nlp.pipe(self.texts, batch_size=128):
            toks = []
            for t in doc:
                if t.is_space or t.is_punct:
                    continue
                if self.remove_stopwords and t.is_stop:
                    continue
                lemma = (t.lemma_ or t.text).lower()
                toks.append(lemma)
            self.lemmas_per_doc.append(toks)

        self._all_lemmas: List[str] = [tok for d in self.lemmas_per_doc for tok in d]

    # ---------- Lexical helpers (now use spaCy lemmas) ----------
    def per_text_ttr(self) -> float:
        """Compute TTR for each text (using lemmas) and return their average."""
        ttrs = []
        for lemmas in self.lemmas_per_doc:
            if len(lemmas) == 0:
                continue
            ttrs.append(len(set(lemmas)) / len(lemmas))
        return float(np.mean(ttrs)) if ttrs else 0.0

    def cumulative_ttr(self) -> float:
        """Treat the dataset as one long text and compute TTR on lemmas."""
        tokens = self._all_lemmas
        return (len(set(tokens)) / len(tokens)) if tokens else 0.0

    def msttr(self, segment_size: int = 100) -> float:
        """
        Mean Segmental TTR (MSTTR) on the concatenated lemma sequence.
        """
        tokens = self._all_lemmas
        if len(tokens) < segment_size or segment_size <= 0:
            return 0.0
        n_segments = len(tokens) // segment_size
        if n_segments == 0:
            return 0.0
        ttrs = []
        for i in range(n_segments):
            seg = tokens[i * segment_size : (i + 1) * segment_size]
            ttrs.append(len(set(seg)) / len(seg))
        return float(np.mean(ttrs)) if ttrs else 0.0

    def distinct_n(self, n: int = 2) -> float:
        """Corpus-level distinct-n over lemmas."""
        tokens = self._all_lemmas
        if n <= 0 or len(tokens) < n:
            return float("nan")
        # sliding window n-grams
        total = len(tokens) - n + 1
        ngrams = set(tuple(tokens[i:i+n]) for i in range(total))
        return len(ngrams) / total if total > 0 else float("nan")

    def mtld(self, threshold: float = 0.72) -> float:
        """Two-pass MTLD on lemmas (forward and backward)."""
        toks = self._all_lemmas
        if not toks:
            return float("nan")

        def mtld_pass(tokens):
            factors, start, types = 0, 0, set()
            for i, tok in enumerate(tokens, 1):
                types.add(tok)
                if len(types) / i < threshold:
                    factors += 1
                    start, types = i, set()
            # partial factor for remainder
            excess = len(tokens) - start
            partial = (excess / max(1, len(types))) if len(types) > 0 else 0.0
            denom = factors + partial
            return len(tokens) / denom if denom > 0 else float("inf")

        return float((mtld_pass(toks) + mtld_pass(list(reversed(toks)))) / 2.0)

    def yule_k(self) -> float:
        toks = self._all_lemmas
        N = len(toks)
        if N == 0:
            return float("nan")
        freqs = Counter(toks)
        M1 = N
        M2 = sum(f * f for f in freqs.values())
        return 1e4 * (M2 - M1) / (M1 * M1)

    def compression_ratio(self) -> float:
        text_concat = " ".join(self.texts).encode("utf-8")
        if len(text_concat) == 0:
            return float("nan")
        comp = zlib.compress(text_concat)
        return len(comp) / len(text_concat)

    # ---------- Embedding-based ----------
    def avg_cosine_distance(self) -> float:
        """
        Average pairwise cosine distance across all embeddings.
        For normalized embeddings, cosine similarity = X @ X.T; distance = 1 - sim.
        """
        X = self.embeddings
        n = len(X)
        if n < 2:
            return float("nan")
        K = X @ X.T
        iu = np.triu_indices(n, k=1)
        dists = 1.0 - K[iu]
        return float(np.mean(dists)) if dists.size else float("nan")

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
            K = X @ X.T
        elif kernel == "rbf":
            sq = np.sum(X**2, axis=1, keepdims=True)
            D2 = sq + sq.T - 2 * (X @ X.T)
            if rbf_sigma is None:
                tri = D2[np.triu_indices(n, k=1)]
                med = np.median(tri) if tri.size else 1.0
                rbf_sigma = np.sqrt(max(med, 1e-12) / 2.0)
            K = np.exp(-D2 / (2.0 * rbf_sigma**2))
        else:
            raise ValueError("kernel must be 'cosine' or 'rbf'")

        Z = K / max(tau, 1e-12)
        Z = Z - Z.max(axis=1, keepdims=True)  # numerical stability
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True) + 1e-12
        return float(np.trace(P))

    def vendi(self, kernel: str = "rbf", rbf_sigma: Optional[float] = None) -> float:
        """
        Vendi via vendi-score package; builds a PSD similarity matrix K.
        """
        X = self.embeddings
        n = len(X)
        if n == 0:
            return float("nan")

        if kernel == "cosine":
            # Map cosine [-1,1] -> [0,1], then symmetrize
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

        K = 0.5 * (S + S.T)  # numerical symmetrization
        return float(vendi.score_K(K))

    # ---------- Run all ----------
    def compute_all(self, k_for_silhouette: int = 10, dump_file: bool = False) -> Dict[str, Any]:
        scores = {
            "silhouette": self.silhouette(k_for_silhouette),
            "dcs_score": self.dcs(tau=0.07, kernel="cosine"),
            "vendi_score": self.vendi(),
            "per_text_ttr": self.per_text_ttr(),
            "cumulative_ttr": self.cumulative_ttr(),
            "msttr100": self.msttr(),
            "compression_ratio": self.compression_ratio(),
            "yule_k": self.yule_k(),
            "mtld": self.mtld(),
            "distinct_1": self.distinct_n(1),
            "distinct_2": self.distinct_n(2),
            "distinct_3": self.distinct_n(3),
            "avg_cosine_distance": self.avg_cosine_distance(),
        }

        if dump_file:
            with open("gradio_demos/personas_viewer/scores.json", "w") as f_out:
                json.dump(scores, f_out, indent=2)

        return scores


# ---------------- Example ----------------
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast dark-colored fox leaps above a sleeping canine.",
        "Transformers are powerful models for language tasks.",
        "Neural networks can learn complex representations of text."
    ]
    dm = DiversityMetrics(texts, remove_stopwords=False)
    print(dm.compute_all(k_for_silhouette=3))