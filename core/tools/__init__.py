"""
Tools package for the RAG agent.
Exports all available tools and a factory function.
"""
from __future__ import annotations

from typing import Any, Optional


def get_all_tools(
    user_id: str,
    pinecone_manager: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
) -> list:
    """Build and return all agent tools with the required dependencies injected.

    Args:
        user_id: Current user/session identifier.
        pinecone_manager: PineconeManager instance (may be None).
        embedding_model: EmbeddingModel instance (may be None).

    Returns:
        List of LangChain tool callables ready to be bound to the LLM.
    """
    from core.tools.search_documents import make_search_tool
    from core.tools.summarize_document import make_summarize_tool
    from core.tools.analyze_csv import make_analyze_csv_tool
    from core.tools.calculator import calculadora
    from core.tools.list_documents import make_list_documents_tool

    tools = [
        make_search_tool(
            user_id=user_id,
            pinecone_manager=pinecone_manager,
            embedding_model=embedding_model,
        ),
        make_summarize_tool(
            user_id=user_id,
            pinecone_manager=pinecone_manager,
            embedding_model=embedding_model,
        ),
        make_analyze_csv_tool(
            user_id=user_id,
            pinecone_manager=pinecone_manager,
            embedding_model=embedding_model,
        ),
        calculadora,
        make_list_documents_tool(
            user_id=user_id,
            pinecone_manager=pinecone_manager,
        ),
    ]
    return tools
