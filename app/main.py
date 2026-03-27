"""
Main Streamlit application for AgenteRag.
Provides a chat interface for RAG-powered document Q&A.
"""
import streamlit as st
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.session_manager import SessionManager
from processors import get_processor
from utils.chunking import chunk_document


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgenteRag",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedding_model():
    """Load and cache the SentenceTransformer embedding model."""
    try:
        from vectorstore.embeddings import EmbeddingModel
        settings = get_settings()
        return EmbeddingModel(model_name=settings.embedding_model)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo de embeddings: {e}")
        return None


@st.cache_resource(show_spinner="Conectando a Pinecone...")
def load_pinecone_manager():
    """Load and cache the Pinecone manager."""
    try:
        from vectorstore.pinecone_manager import PineconeManager
        settings = get_settings()
        if not settings.pinecone_api_key or settings.pinecone_api_key == "tu_clave_pinecone_aqui":
            return None
        return PineconeManager(settings=settings)
    except Exception as e:
        st.warning(f"No se pudo conectar a Pinecone: {e}")
        return None


@st.cache_resource(show_spinner="Inicializando agente...")
def load_agent(user_id: str):
    """Load and cache the RAG agent for the given user."""
    try:
        from core.agent import RAGAgent
        settings = get_settings()
        if not settings.groq_api_key or settings.groq_api_key == "gsk_tu_clave_aqui":
            return None
        return RAGAgent(settings=settings, user_id=user_id)
    except Exception as e:
        st.warning(f"No se pudo inicializar el agente: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: check service status
# ---------------------------------------------------------------------------

def check_groq_status() -> bool:
    settings = get_settings()
    return bool(settings.groq_api_key and settings.groq_api_key != "gsk_tu_clave_aqui")


def check_pinecone_status() -> bool:
    settings = get_settings()
    return bool(settings.pinecone_api_key and settings.pinecone_api_key != "tu_clave_pinecone_aqui")


# ---------------------------------------------------------------------------
# File processing helper
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt", "csv", "xlsx", "xls", "md", "log"]


def process_uploaded_file(uploaded_file, user_id: str) -> bool:
    """Process an uploaded file, chunk it and upsert to Pinecone.

    Returns True on success, False on failure.
    """
    embedding_model = load_embedding_model()
    pinecone_manager = load_pinecone_manager()

    if embedding_model is None:
        st.error("El modelo de embeddings no está disponible.")
        return False

    if pinecone_manager is None:
        st.error("Pinecone no está disponible. Verifica tu API key.")
        return False

    try:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        extension = filename.rsplit(".", 1)[-1].lower()

        processor = get_processor(extension)
        if processor is None:
            st.error(f"Tipo de archivo no soportado: .{extension}")
            return False

        text = processor.process(file_bytes)
        if not text or not text.strip():
            st.warning(f"No se pudo extraer texto de {filename}.")
            return False

        settings = get_settings()
        chunks = chunk_document(
            text=text,
            filename=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            st.warning(f"No se generaron chunks para {filename}.")
            return False

        pinecone_manager.upsert_chunks(
            chunks=chunks,
            user_id=user_id,
            embedding_model=embedding_model,
        )

        SessionManager.add_document(filename, {
            "chunks": len(chunks),
            "extension": extension,
            "size_bytes": len(file_bytes),
        })

        st.success(f"'{filename}' indexado correctamente ({len(chunks)} chunks).")
        return True

    except Exception as e:
        st.error(f"Error procesando '{uploaded_file.name}': {e}")
        return False


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # Initialize session state
    SessionManager.init_session()
    user_id = SessionManager.get_user_id()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("🤖 AgenteRag")
        st.markdown("---")

        # API status indicators
        st.subheader("Estado de servicios")
        groq_ok = check_groq_status()
        pinecone_ok = check_pinecone_status()

        if groq_ok:
            st.markdown("🟢 **Groq** — OK")
        else:
            st.markdown("🔴 **Groq** — Sin API key")

        if pinecone_ok:
            st.markdown("🟢 **Pinecone** — OK")
        else:
            st.markdown("🔴 **Pinecone** — Sin API key")

        st.markdown("---")

        # File uploader
        st.subheader("Subir documentos")
        uploaded_files = st.file_uploader(
            "Selecciona archivos",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                existing_docs = SessionManager.get_documents()
                if filename not in existing_docs:
                    with st.spinner(f"Procesando {filename}..."):
                        process_uploaded_file(uploaded_file, user_id)

        st.markdown("---")

        # Indexed documents list
        st.subheader("Documentos indexados")
        documents = SessionManager.get_documents()
        if documents:
            for doc_name, meta in documents.items():
                st.markdown(
                    f"📄 **{doc_name}**  \n"
                    f"  Chunks: {meta.get('chunks', '?')} | "
                    f"Tipo: {meta.get('extension', '?').upper()}"
                )
        else:
            st.info("No hay documentos indexados aún.")

        st.markdown("---")
        st.caption(f"Session ID: `{user_id[:8]}...`")

    # -----------------------------------------------------------------------
    # Main chat interface
    # -----------------------------------------------------------------------
    st.title("💬 Chat con tus documentos")

    # Display chat history
    messages = SessionManager.get_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Escribe tu pregunta aquí...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        SessionManager.add_message("user", user_input)

        # Load agent and get response
        agent = load_agent(user_id)

        if agent is None:
            error_msg = (
                "El agente no está disponible. Verifica que la variable "
                "`GROQ_API_KEY` esté configurada en tu archivo `.env`."
            )
            with st.chat_message("assistant"):
                st.error(error_msg)
            SessionManager.add_message("assistant", error_msg)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    try:
                        response = agent.chat(user_input)
                        st.markdown(response)
                        SessionManager.add_message("assistant", response)
                    except Exception as e:
                        error_msg = f"Error al procesar tu consulta: {e}"
                        st.error(error_msg)
                        SessionManager.add_message("assistant", error_msg)


if __name__ == "__main__":
    main()
