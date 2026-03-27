"""
LangChain tool for analyzing CSV/Excel data stored in Pinecone metadata.
"""
from __future__ import annotations

import io
from typing import Any, Optional

from langchain_core.tools import tool


def make_analyze_csv_tool(
    user_id: str,
    pinecone_manager: Optional[Any],
    embedding_model: Optional[Any],
):
    @tool
    def analizar_datos_csv(document_name: str) -> str:
        """Analiza los datos de un archivo CSV o Excel indexado.

        Recupera el contenido del archivo desde Pinecone y calcula estadísticas
        básicas: dimensiones, tipos de columna, valores nulos y estadísticas
        descriptivas numéricas.

        Args:
            document_name: Nombre del archivo CSV o Excel (ej. 'datos.csv').

        Returns:
            Informe de estadísticas del dataset en texto plano.
        """
        if pinecone_manager is None or embedding_model is None:
            return (
                "El servicio de análisis no está disponible. "
                "Verifica que Pinecone y el modelo de embeddings estén configurados."
            )

        try:
            import pandas as pd

            results = pinecone_manager.search(
                query=document_name,
                user_id=user_id,
                embedding_model=embedding_model,
                top_k=50,
            )

            doc_chunks = [
                r for r in results
                if r.get("metadata", {}).get("filename", "").lower()
                == document_name.lower()
            ]

            if not doc_chunks:
                doc_chunks = results

            doc_chunks.sort(
                key=lambda r: r.get("metadata", {}).get("chunk_index", 0)
            )

            csv_parts = [
                r.get("metadata", {}).get("text", "")
                for r in doc_chunks
                if r.get("metadata", {}).get("text")
            ]

            if not csv_parts:
                return f"No se encontró contenido para '{document_name}'."

            combined_text = "\n".join(csv_parts)

            try:
                df = pd.read_csv(io.StringIO(combined_text))
            except Exception:
                return (
                    f"No se pudo parsear '{document_name}' como CSV.\n"
                    f"Contenido recuperado:\n{combined_text[:500]}"
                )

            lines = [
                f"Análisis de '{document_name}'",
                f"{'=' * 40}",
                f"Filas:    {df.shape[0]}",
                f"Columnas: {df.shape[1]}",
                "",
                "Tipos de datos:",
            ]

            for col, dtype in df.dtypes.items():
                null_count = df[col].isnull().sum()
                lines.append(f"  {col}: {dtype} ({null_count} nulos)")

            lines.append("")
            lines.append("Estadísticas descriptivas (columnas numéricas):")

            numeric_desc = df.describe()
            if numeric_desc.empty:
                lines.append("  No hay columnas numéricas.")
            else:
                lines.append(numeric_desc.to_string())

            return "\n".join(lines)

        except Exception as e:
            return f"Error al analizar '{document_name}': {e}"

    return analizar_datos_csv
