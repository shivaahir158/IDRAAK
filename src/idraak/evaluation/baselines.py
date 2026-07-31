"""Baseline drift detection methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from idraak.utils.logging import get_logger

logger = get_logger("baselines")


class ExactMatchBaseline:
    """Baseline 1: Exact string match."""

    def predict(self, text_a: str, text_b: str) -> dict[str, Any]:
        match = text_a.strip().lower() == text_b.strip().lower()
        return {
            "drift_detected": not match,
            "confidence": 1.0 if not match else 0.0,
            "method": "exact_match",
        }


class TokenOverlapBaseline:
    """Baseline 2: Jaccard token overlap."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def predict(self, text_a: str, text_b: str) -> dict[str, Any]:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return {"drift_detected": True, "confidence": 1.0, "similarity": 0.0, "method": "token_overlap"}

        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        drift = jaccard < self.threshold
        return {
            "drift_detected": drift,
            "confidence": 1.0 - jaccard if drift else jaccard,
            "similarity": jaccard,
            "method": "token_overlap",
        }


class EmbeddingBaseline:
    """Baseline 3: Multilingual sentence embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        threshold: float = 0.85,
    ):
        self.threshold = threshold
        self._model = None
        self._model_name = model_name

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded embedding model: {self._model_name}")
            except ImportError:
                logger.warning("sentence-transformers not installed; using random embeddings")
                self._model = "mock"
        return self._model

    def predict(self, text_a: str, text_b: str) -> dict[str, Any]:
        model = self._load_model()

        if model == "mock":
            # Deterministic mock based on text similarity
            overlap = len(set(text_a.lower().split()) & set(text_b.lower().split()))
            total = max(len(set(text_a.lower().split()) | set(text_b.lower().split())), 1)
            similarity = overlap / total
        else:
            embeddings = model.encode([text_a, text_b])
            similarity = float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
            ))

        drift = similarity < self.threshold
        return {
            "drift_detected": drift,
            "confidence": 1.0 - similarity if drift else similarity,
            "similarity": similarity,
            "method": "embedding",
            "model": self._model_name,
        }

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts."""
        model = self._load_model()
        if model == "mock":
            return np.random.randn(len(texts), 768).astype(np.float32)
        return model.encode(texts)
