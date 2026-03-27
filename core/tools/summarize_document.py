"""
LangChain tool for retrieving and summarizing a full document from Pinecone.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool


def make_summarize_tool(
    user_id: str,
    pinecone_manager: Optional[Any],
    embedding_model: Optional[Any],
):
    """Factory that creates a summarize-document tool with injected dependencies.

    Args:
        user_id: Namespace identifier for Pinecone queries.
        pinecone_manager: PineconeManager instance.
        embedding_model: EmbeddingModel instance.

    Returns:
        A LangChain @tool callable.
    """

    @tool
    def resumir_documento(document_name: str) -> str:
        """Recupera y muestra el contenido completo de un documento indexado.

        Usa esta herramienta cuando el usuario pida un resumen o quiera conocer
        el contenido general de un documento específico.

        Args:
            document_name: Nombre exacto del archivo (ej. 'reporte.pdf').

        Returns:
            Texto unificado con todos los fragmentos del documento.
        """
        if pinecone_manager is None or embedding_model is None:
            return (
                "El servicio de recuperación de documentos no está disponible. "
                "Verifica que Pinecone y el modelo de embeddings estén configurados."
            )

        try:
            # Search with the document name as query to retrieve its chunks
            results = pinecone_manager.search(
                query=document_name,
                user_id=user_id,
                embedding_model=embedding_model,
                top_k=50,  # Retrieve many chunks for a full summary
            )

            if not results:
                return f"No se encontraron chunks para el documento '{document_name}'."

            # Filter results to only those belonging to the requested document
            doc_chunks = [
                r for r in results
                if r.get("metadata", {}).get("filename", "").lower()
                == document_name.lower()
            ]

            if not doc_chunks:
                # Fallback: show all results if none match exactly
                doc_chunks = results

            # Sort by chunk_index to reconstruct document order
            doc_chunks.sort(
                key=lambda r: r.get("metadata", {}).get("chunk_index", 0)
            )

            texts = [
                r.get("metadata", {}).get("text", "")
                for r in doc_chunks
                if r.get("metadata", {}).get("text")
            ]

            if not texts:
                return f"No se pudo recuperar el texto del documento '{document_name}'."

            full_text = "\n\n".join(texts)
            return (
                f"Contenido del documento '{document_name}' "
                f"({len(doc_chunks)} fragmentos recuperados):\n\n{full_text}"
            )

        except Exception as e:
            return f"Error al recuperar el documento '{document_name}': {e}"

    return resumir_documento
