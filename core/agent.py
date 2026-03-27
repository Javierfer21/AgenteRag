"""
LangGraph-based RAG agent.
Orchestrates LLM reasoning with tool use for document retrieval and analysis.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from core.state import AgentState
from core.memory import ConversationMemory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres AgenteRag, un asistente inteligente especializado en análisis de documentos.
Tienes acceso a las siguientes herramientas:
- buscar_en_documentos: busca información relevante en los documentos indexados
- resumir_documento: genera un resumen de un documento específico
- analizar_datos_csv: analiza datos de archivos CSV/Excel con estadísticas
- calculadora: realiza cálculos matemáticos de forma segura
- listar_documentos: muestra todos los documentos disponibles

Responde siempre en el mismo idioma que el usuario.
Cuando el usuario pregunte sobre documentos, utiliza las herramientas disponibles.
Sé conciso, preciso y útil."""


def _should_continue(state: AgentState) -> str:
    """Routing function: continue to tools if the last message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "nodo_herramientas"
    return END


class RAGAgent:
    """LangGraph agent that combines LLM reasoning with RAG tools."""

    def __init__(self, settings: Any, user_id: str) -> None:
        """Set up LLM, tools, memory and build the LangGraph graph.

        Args:
            settings: Application settings (Settings instance).
            user_id: Unique identifier for the current user/session.
        """
        self.settings = settings
        self.user_id = user_id

        # Initialize memory
        self.memory = ConversationMemory(
            db_path=settings.sqlite_db_path,
            user_id=user_id,
        )

        # Build tools with access to pinecone/embedding via closures
        self.tools = self._build_tools()

        # Initialize LLM and bind tools
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the LangGraph graph
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list:
        """Instantiate and return all agent tools."""
        from core.tools import get_all_tools
        from vectorstore.embeddings import EmbeddingModel
        from vectorstore.pinecone_manager import PineconeManager

        try:
            embedding_model = EmbeddingModel(model_name=self.settings.embedding_model)
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            embedding_model = None

        try:
            if (
                self.settings.pinecone_api_key
                and self.settings.pinecone_api_key != "tu_clave_pinecone_aqui"
            ):
                pinecone_manager = PineconeManager(settings=self.settings)
            else:
                pinecone_manager = None
        except Exception as e:
            logger.warning(f"Could not connect to Pinecone: {e}")
            pinecone_manager = None

        return get_all_tools(
            user_id=self.user_id,
            pinecone_manager=pinecone_manager,
            embedding_model=embedding_model,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Build and compile the LangGraph StateGraph."""
        tool_node = ToolNode(self.tools)

        def nodo_agente(state: AgentState) -> dict:
            """Agent node: calls the LLM with tool bindings."""
            messages = state["messages"]
            # Prepend system message if it's the first call
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("nodo_agente", nodo_agente)
        graph_builder.add_node("nodo_herramientas", tool_node)

        graph_builder.set_entry_point("nodo_agente")

        graph_builder.add_conditional_edges(
            "nodo_agente",
            _should_continue,
            {
                "nodo_herramientas": "nodo_herramientas",
                END: END,
            },
        )
        graph_builder.add_edge("nodo_herramientas", "nodo_agente")

        return graph_builder.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, message: str) -> str:
        """Send a message to the agent and return the text response.

        Args:
            message: User's input message.

        Returns:
            Agent's response as a plain string.
        """
        # Persist user message
        self.memory.save_message("user", message)

        # Build initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "user_id": self.user_id,
            "documents": [],
        }

        try:
            result = self.graph.invoke(initial_state)
            # Extract the last assistant message
            response_text = ""
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and not hasattr(msg, "tool_calls"):
                    response_text = msg.content
                    break
                # Also check for AIMessage with no tool calls
                if (
                    hasattr(msg, "content")
                    and hasattr(msg, "tool_calls")
                    and not msg.tool_calls
                ):
                    response_text = msg.content
                    break

            if not response_text:
                response_text = result["messages"][-1].content

        except Exception as e:
            logger.error(f"Agent error: {e}")
            raise

        # Persist assistant response
        self.memory.save_message("assistant", response_text)
        return response_text
