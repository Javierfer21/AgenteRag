"""
Pinecone vector store manager.
Handles index creation, upsert, search, listing and deletion of document vectors.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Maximum vectors per upsert batch (Pinecone limit)
_UPSERT_BATCH_SIZE = 100


class PineconeManager:
    """Manages interactions with a Pinecone vector index."""

    def __init__(self, settings: Any) -> None:
        """Initialize the Pinecone client and ensure the index exists.

        Args:
            settings: Application Settings instance with Pinecone credentials.

        Raises:
            ImportError: If pinecone-client is not installed.
            Exception: If the Pinecone client cannot connect.
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise ImportError(
                "pinecone-client is required. "
                "Install it with: pip install pinecone-client"
            ) from exc

        self.settings = settings
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.embedding_dimension

        logger.info("Initializing Pinecone client...")
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._get_or_create_index(ServerlessSpec)
        logger.info(f"Pinecone index '{self.index_name}' is ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_index(self, ServerlessSpec: Any) -> Any:
        """Return existing index or create a new one.

        Args:
            ServerlessSpec: Pinecone ServerlessSpec class.

        Returns:
            Pinecone Index object.
        """
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(
                f"Index '{self.index_name}' not found. Creating with "
                f"dimension={self.dimension}, metric='cosine'."
            )
            self._pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment,
                ),
            )
            # Wait for the index to be ready
            self._wait_for_index_ready()
        else:
            logger.info(f"Index '{self.index_name}' already exists.")

        return self._pc.Index(self.index_name)

    def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Poll until the Pinecone index reports 'Ready' status.

        Args:
            timeout: Maximum seconds to wait before raising.

        Raises:
            TimeoutError: If the index is not ready within timeout seconds.
        """
        start = time.time()
        while time.time() - start < timeout:
            description = self._pc.describe_index(self.index_name)
            status = description.get("status", {})
            if status.get("ready", False):
                return
            logger.debug("Waiting for index to be ready...")
            time.sleep(2)
        raise TimeoutError(
            f"Pinecone index '{self.index_name}' was not ready within {timeout}s."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[dict],
        user_id: str,
        embedding_model: Any,
        doc_metadata: dict | None = None,
    ) -> None:
        """Embed document chunks and upsert them to Pinecone.

        Args:
            chunks: List of chunk dicts with keys: text, filename, chunk_index, chunk_id.
            user_id: Pinecone namespace to use for this user's data.
            embedding_model: EmbeddingModel instance for generating vectors.
            doc_metadata: Optional extra metadata (size_bytes, char_count, etc.)
                          stored in every chunk so it can be retrieved later.
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        vectors = embedding_model.embed_batch(texts)

        extra = doc_metadata or {}
        pinecone_vectors = []
        for chunk, vector in zip(chunks, vectors):
            pinecone_vectors.append({
                "id": chunk["chunk_id"],
                "values": vector,
                "metadata": {
                    "text": chunk["text"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    **extra,
                },
            })

        # Upsert in batches
        for i in range(0, len(pinecone_vectors), _UPSERT_BATCH_SIZE):
            batch = pinecone_vectors[i : i + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=user_id)
            logger.debug(f"Upserted batch {i // _UPSERT_BATCH_SIZE + 1} ({len(batch)} vectors).")

    def search(
        self,
        query: str,
        user_id: str,
        embedding_model: Any,
        top_k: int = 5,
    ) -> list[dict]:
        """Perform a semantic similarity search.

        Args:
            query: Natural language query string.
            user_id: Pinecone namespace to search within.
            embedding_model: EmbeddingModel instance for query embedding.
            top_k: Number of top results to return.

        Returns:
            List of result dicts with keys: id, score, metadata.
        """
        query_vector = embedding_model.embed(query)
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=user_id,
            include_metadata=True,
        )

        results = []
        for match in response.get("matches", []):
            results.append({
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "metadata": match.get("metadata", {}),
            })
        return results

    def list_documents(self, user_id: str) -> list[str]:
        """List all unique document filenames in a user's namespace.

        Uses a dummy query to retrieve vectors and extract unique filenames
        from metadata (Pinecone does not support native metadata-only listing).

        Args:
            user_id: Pinecone namespace identifier.

        Returns:
            Sorted list of unique filename strings.
        """
        try:
            # Query with a zero vector to retrieve all vectors' metadata
            dummy_vector = [0.0] * self.dimension
            response = self._index.query(
                vector=dummy_vector,
                top_k=1000,
                namespace=user_id,
                include_metadata=True,
            )
            filenames: set[str] = set()
            for match in response.get("matches", []):
                filename = match.get("metadata", {}).get("filename")
                if filename:
                    filenames.add(filename)
            return sorted(filenames)
        except Exception as e:
            logger.warning(f"Could not list documents for user '{user_id}': {e}")
            return []

    def get_documents_metadata(self, user_id: str) -> dict[str, dict]:
        """Return a dict of {filename: metadata} for all indexed documents.

        Aggregates chunk counts and document-level metadata stored inside
        the vector metadata fields (size_bytes, char_count, etc.).

        Args:
            user_id: Pinecone namespace identifier.

        Returns:
            Dict mapping filename -> metadata dict with keys:
            chunks, size_bytes, char_count, extension.
        """
        try:
            dummy_vector = [0.0] * self.dimension
            response = self._index.query(
                vector=dummy_vector,
                top_k=1000,
                namespace=user_id,
                include_metadata=True,
            )
            docs: dict[str, dict] = {}
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                filename = meta.get("filename")
                if not filename:
                    continue
                if filename not in docs:
                    docs[filename] = {
                        "chunks": 0,
                        "extension": filename.rsplit(".", 1)[-1].lower() if "." in filename else "",
                        "size_bytes": int(meta.get("size_bytes", 0)),
                        "char_count": int(meta.get("char_count", 0)),
                    }
                docs[filename]["chunks"] += 1
            return docs
        except Exception as e:
            logger.warning(f"Could not get document metadata for user '{user_id}': {e}")
            return {}

    def delete_document(self, filename: str, user_id: str) -> None:
        """Delete ALL vectors associated with a specific document filename.

        Uses index.list() to paginate through every vector ID in the namespace
        (not limited by top_k), then fetches metadata in batches to identify
        vectors belonging to the target file, and deletes them all.

        Args:
            filename: The document filename whose vectors should be removed.
            user_id: Pinecone namespace identifier.
        """
        try:
            ids_to_delete: list[str] = []

            # list() paginates through ALL vector IDs in the namespace
            for id_batch in self._index.list(namespace=user_id):
                if not id_batch:
                    continue
                fetch_response = self._index.fetch(ids=id_batch, namespace=user_id)
                for vec_id, vec_data in fetch_response.vectors.items():
                    meta = getattr(vec_data, "metadata", {}) or {}
                    if meta.get("filename") == filename:
                        ids_to_delete.append(vec_id)

            if not ids_to_delete:
                logger.info(f"No vectors found for '{filename}' in namespace '{user_id}'.")
                return

            # Delete in batches of 1000 (Pinecone hard limit per request)
            for i in range(0, len(ids_to_delete), _UPSERT_BATCH_SIZE):
                batch = ids_to_delete[i : i + _UPSERT_BATCH_SIZE]
                self._index.delete(ids=batch, namespace=user_id)

            logger.info(
                f"Deleted {len(ids_to_delete)} vectors for '{filename}' "
                f"in namespace '{user_id}'."
            )
        except Exception as e:
            logger.error(f"Error deleting document '{filename}': {e}")
            raise
