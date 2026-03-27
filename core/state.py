"""
LangGraph agent state definition.
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all nodes in the LangGraph agent graph."""

    # Conversation messages — uses add_messages reducer to append new messages
    messages: Annotated[list, add_messages]

    # Unique identifier for the user/session
    user_id: str

    # List of document filenames available for retrieval
    documents: list[str]
