"""
Streamlit session state manager.
Handles all session-related state initialization and access.
"""
import uuid
import streamlit as st
from typing import Any


class SessionManager:
    """Manages Streamlit session state for the RAG application."""

    # Keys used in session state
    USER_ID_KEY = "user_id"
    MESSAGES_KEY = "messages"
    DOCUMENTS_KEY = "documents"
    AGENT_KEY = "agent"
    PINECONE_MANAGER_KEY = "pinecone_manager"
    EMBEDDING_MODEL_KEY = "embedding_model"

    @classmethod
    def init_session(cls) -> None:
        """Initialize all session state keys with default values.

        The user_id is persisted in the URL query params (?uid=...) so that
        refreshing the page keeps the same Pinecone namespace and documents.
        """
        if cls.USER_ID_KEY not in st.session_state:
            uid = st.query_params.get("uid")
            if not uid:
                uid = str(uuid.uuid4())
                st.query_params["uid"] = uid
            st.session_state[cls.USER_ID_KEY] = uid

        if cls.MESSAGES_KEY not in st.session_state:
            st.session_state[cls.MESSAGES_KEY] = []

        if cls.DOCUMENTS_KEY not in st.session_state:
            st.session_state[cls.DOCUMENTS_KEY] = {}

        if cls.AGENT_KEY not in st.session_state:
            st.session_state[cls.AGENT_KEY] = None

        if cls.PINECONE_MANAGER_KEY not in st.session_state:
            st.session_state[cls.PINECONE_MANAGER_KEY] = None

        if cls.EMBEDDING_MODEL_KEY not in st.session_state:
            st.session_state[cls.EMBEDDING_MODEL_KEY] = None

    @classmethod
    def get_user_id(cls) -> str:
        """Return the current session user ID."""
        cls.init_session()
        return st.session_state[cls.USER_ID_KEY]

    @classmethod
    def add_message(cls, role: str, content: str) -> None:
        """Add a message to the session message history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message text content
        """
        cls.init_session()
        st.session_state[cls.MESSAGES_KEY].append({
            "role": role,
            "content": content
        })

    @classmethod
    def get_messages(cls) -> list:
        """Return the list of messages in the current session."""
        cls.init_session()
        return st.session_state[cls.MESSAGES_KEY]

    @classmethod
    def add_document(cls, name: str, metadata: dict) -> None:
        """Track an uploaded document in session state.

        Args:
            name: Document filename
            metadata: Additional document metadata (e.g. chunks, size)
        """
        cls.init_session()
        st.session_state[cls.DOCUMENTS_KEY][name] = metadata

    @classmethod
    def get_documents(cls) -> dict:
        """Return the dictionary of tracked documents."""
        cls.init_session()
        return st.session_state[cls.DOCUMENTS_KEY]

    @classmethod
    def get_agent(cls) -> Any:
        """Return the current agent instance."""
        cls.init_session()
        return st.session_state[cls.AGENT_KEY]

    @classmethod
    def set_agent(cls, agent: Any) -> None:
        """Set the agent instance in session state."""
        cls.init_session()
        st.session_state[cls.AGENT_KEY] = agent
