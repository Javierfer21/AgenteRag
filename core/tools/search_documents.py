"""
LangChain tool for semantic search across indexed documents.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool


def make_search_tool(
    user_id: str,
    pinecone_manager: Optional[Any],
    embedding_model: Optional[Any],
):
    """Factory that creates a closure-based search tool with injected dependencies.

    Args:
        user_id: Namespace identifier for Pinecone queries.
        pinecone_manager: PineconeManager instance for vector search.
        embedding_model: EmbeddingModel instance for query embedding.

    Returns:
        A LangChain @tool callable.
    """

    @tool
    def buscar_en_documentos(query: str) -> str:
        """Busca información relevante en los documentos indexados del usuario.

        Usa esta herramienta cuando el usuario pregunte sobre el contenido de
        sus documentos o necesite encontrar información específica.

        Args:
            query: La pregunta o términos de búsqueda en lenguaje natural.

        Returns:
            Fragmentos de documentos relevantes con su fuente y puntuación.
        """
        if pinecone_manager is None or embedding_model is None:
            return (
                "El servicio de búsqueda no está disponible. "
                "Verifica que Pinecone y el modelo de embeddings estén configurados."
            )

        try:
            results = pinecone_manager.search(
                query=query,
                user_id=user_id,
                embedding_model=embedding_model,
                top_k=5,
            )

            if not results:
                return "No se encontraron documentos relevantes para tu consulta."

            output_parts = [f"Resultados de búsqueda para: '{query}'\n"]
            for i, result in enumerate(results, start=1):
                metadata = result.get("metadata", {})
                filename = metadata.get("filename", "Desconocido")
                chunk_index = metadata.get("chunk_index", "?")
                text = metadata.get("text", "")
                score = result.get("score", 0.0)

                output_parts.append(
                    f"--- Resultado {i} ---\n"
                    f"Documento: {filename} (chunk {chunk_index})\n"
                    f"Relevancia: {score:.3f}\n"
                    f"Contenido:\n{text}\n"
                )

            return "\n".join(output_parts)

        except Exception as e:
            return f"Error al realizar la búsqueda: {e}"

    return buscar_en_documentos
