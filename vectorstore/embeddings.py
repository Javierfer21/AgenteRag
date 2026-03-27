"""
Embedding model wrapper around SentenceTransformers.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a SentenceTransformer model for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the SentenceTransformer model.

        Args:
            model_name: HuggingFace model identifier to load.
                        Defaults to 'all-MiniLM-L6-v2' (384 dimensions).

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [v.tolist() for v in vectors]
