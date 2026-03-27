"""
LangChain tool for listing all indexed documents for a user.
"""
from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool


def make_list_documents_tool(
    user_id: str,
    pinecone_manager: Optional[Any],
):
    """Factory that creates a list-documents tool with injected dependencies.

    Args:
        user_id: Namespace identifier for Pinecone queries.
        pinecone_manager: PineconeManager instance.

    Returns:
        A LangChain @tool callable.
    """

    @tool
    def listar_documentos() -> str:
        """Lista todos los documentos indexados disponibles para el usuario actual.

        Usa esta herramienta cuando el usuario pregunte qué documentos tiene
        disponibles o quiera conocer los archivos que puede consultar.

        Returns:
            Lista formateada de nombres de archivo indexados.
        """
        if pinecone_manager is None:
            return (
                "El servicio de listado no está disponible. "
                "Verifica que Pinecone esté configurado correctamente."
            )

        try:
            document_names = pinecone_manager.list_documents(user_id=user_id)

            if not document_names:
                return (
                    "No hay documentos indexados en esta sesión. "
                    "Por favor, sube algunos archivos primero."
                )

            lines = [
                f"Documentos disponibles ({len(document_names)} en total):",
                "",
            ]
            for i, name in enumerate(sorted(document_names), start=1):
                lines.append(f"  {i}. {name}")

            return "\n".join(lines)

        except Exception as e:
            return f"Error al listar documentos: {e}"

    return listar_documentos
