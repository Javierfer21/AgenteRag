"""
Main Streamlit application for AgenteRag.
Provides a chat interface for RAG-powered document Q&A.
"""
import re
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
    try:
        from vectorstore.embeddings import EmbeddingModel
        settings = get_settings()
        return EmbeddingModel(model_name=settings.embedding_model)
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo de embeddings: {e}")
        return None


@st.cache_resource(show_spinner="Conectando a Pinecone...")
def load_pinecone_manager():
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

EXTENSION_ICONS = {
    "pdf": "📕", "docx": "📘", "doc": "📘",
    "txt": "📄", "md": "📄", "log": "📄",
    "csv": "📊", "xlsx": "📊", "xls": "📊", "xlsm": "📊",
}


def process_uploaded_file(uploaded_file, user_id: str) -> dict | None:
    """Process an uploaded file, chunk it and upsert to Pinecone.

    Returns a dict with processing stats on success, None on failure.
    """
    embedding_model = load_embedding_model()
    pinecone_manager = load_pinecone_manager()

    if embedding_model is None:
        st.error("El modelo de embeddings no está disponible.")
        return None
    if pinecone_manager is None:
        st.error("Pinecone no está disponible. Verifica tu API key.")
        return None

    try:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        extension = filename.rsplit(".", 1)[-1].lower()

        processor = get_processor(extension)
        if processor is None:
            st.error(f"Tipo de archivo no soportado: .{extension}")
            return None

        # --- Step 1: extract text ---
        text = processor.process(file_bytes)
        if not text or not text.strip():
            st.warning(f"No se pudo extraer texto de {filename}.")
            return None

        # --- Step 2: chunk ---
        settings = get_settings()
        chunks = chunk_document(
            text=text,
            filename=filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            st.warning(f"No se generaron chunks para {filename}.")
            return None

        # --- Step 3: embed a sample to get dimension info ---
        sample_embedding = embedding_model.embed(chunks[0]["text"])
        embedding_dim = len(sample_embedding)

        # --- Step 4: upsert to Pinecone (including doc metadata for persistence) ---
        pinecone_manager.upsert_chunks(
            chunks=chunks,
            user_id=user_id,
            embedding_model=embedding_model,
            doc_metadata={
                "size_bytes": len(file_bytes),
                "char_count": len(text),
            },
        )

        stats = {
            "filename": filename,
            "extension": extension,
            "size_bytes": len(file_bytes),
            "char_count": len(text),
            "word_count": len(text.split()),
            "chunk_count": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "embedding_dim": embedding_dim,
            "sample_chunks": [c["text"] for c in chunks[:3]],
            "sample_vector": sample_embedding[:8],
        }

        SessionManager.add_document(filename, {
            "chunks": len(chunks),
            "extension": extension,
            "size_bytes": len(file_bytes),
            "char_count": len(text),
            "chunk_size": settings.chunk_size,
        })

        return stats

    except Exception as e:
        st.error(f"Error procesando '{uploaded_file.name}': {e}")
        return None



# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def load_documents_from_pinecone(user_id: str) -> None:
    """Populate session documents from Pinecone on first load (after refresh)."""
    if st.session_state.get("_docs_loaded"):
        return
    st.session_state["_docs_loaded"] = True

    pinecone_manager = load_pinecone_manager()
    if pinecone_manager is None:
        return

    existing = SessionManager.get_documents()
    persisted = pinecone_manager.get_documents_metadata(user_id)
    for filename, meta in persisted.items():
        if filename not in existing:
            SessionManager.add_document(filename, meta)


def delete_document(filename: str, user_id: str) -> None:
    """Remove a document from Pinecone and session state."""
    pinecone_manager = load_pinecone_manager()
    if pinecone_manager:
        pinecone_manager.delete_document(filename, user_id)
    docs = SessionManager.get_documents()
    docs.pop(filename, None)


def main():
    SessionManager.init_session()
    user_id = SessionManager.get_user_id()
    load_documents_from_pinecone(user_id)

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("🤖 AgenteRag")
        st.markdown("---")

        # API status
        st.subheader("Estado de servicios")
        groq_ok = check_groq_status()
        pinecone_ok = check_pinecone_status()
        st.markdown("🟢 **Groq** — OK" if groq_ok else "🔴 **Groq** — Sin API key")
        st.markdown("🟢 **Pinecone** — OK" if pinecone_ok else "🔴 **Pinecone** — Sin API key")

        settings = get_settings()
        st.markdown("---")
        st.subheader("Configuración activa")
        st.caption(f"Modelo LLM: `{settings.llm_model}`")
        st.caption(f"Embeddings: `{settings.embedding_model}`")
        st.caption(f"Chunk size: `{settings.chunk_size}` | Overlap: `{settings.chunk_overlap}`")

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
                    # Store pending file in session to render report in main area
                    if "pending_reports" not in st.session_state:
                        st.session_state.pending_reports = []
                    with st.spinner(f"Procesando {filename}..."):
                        stats = process_uploaded_file(uploaded_file, user_id)
                        if stats:
                            st.session_state.pending_reports.append(stats)

        st.markdown("---")

        # Indexed documents list
        st.subheader("Documentos indexados")
        documents = SessionManager.get_documents()
        reports = st.session_state.get("pending_reports", [])
        # Merge fresh report data into session documents for richer display
        reports_by_name = {r["filename"]: r for r in reports}

        if documents:
            for doc_name, meta in list(documents.items()):
                icon = EXTENSION_ICONS.get(meta.get("extension", ""), "📄")
                size_kb = meta.get("size_bytes", 0) / 1024
                report = reports_by_name.get(doc_name)
                with st.expander(f"{icon} {doc_name}", expanded=(doc_name in reports_by_name)):
                    st.caption(f"**Tamaño:** {size_kb:.1f} KB &nbsp;·&nbsp; **Tipo:** {meta.get('extension','?').upper()}")
                    if meta.get("char_count"):
                        char_line = f"**Caracteres:** {meta['char_count']:,}"
                        if report:
                            char_line += f" &nbsp;·&nbsp; **Palabras:** {report['word_count']:,}"
                        st.caption(char_line)
                    st.caption(f"**Chunks:** {meta.get('chunks', '?')} &nbsp;·&nbsp; **Chunk size:** {meta.get('chunk_size', settings.chunk_size)} chars")
                    if report:
                        st.caption(f"**Modelo embedding:** `{report['embedding_model']}`")
                        st.caption(f"**Dimensión vector:** {report['embedding_dim']}D")
                        with st.expander("🔍 Preview chunks"):
                            for i, chunk_text in enumerate(report["sample_chunks"], 1):
                                st.caption(f"**Chunk {i}**")
                                st.code(chunk_text[:250] + ("..." if len(chunk_text) > 250 else ""), language=None)
                        with st.expander("🧮 Vector (8 primeros valores)"):
                            vector_str = ", ".join(f"{v:.4f}" for v in report["sample_vector"])
                            st.code(f"[{vector_str}, ...]", language=None)
                    confirming = st.session_state.get(f"confirm_del_{doc_name}", False)
                    if not confirming:
                        if st.button("🗑️ Eliminar", key=f"del_{doc_name}", use_container_width=True):
                            st.session_state[f"confirm_del_{doc_name}"] = True
                            st.rerun()
                    else:
                        st.warning(
                            f"¿Eliminar **{doc_name}**?\n\n"
                            "Se borrarán todos sus datos de Pinecone de forma permanente."
                        )
                        col_yes, col_no = st.columns(2)
                        if col_yes.button("✅ Confirmar", key=f"yes_{doc_name}", use_container_width=True):
                            st.session_state.pop(f"confirm_del_{doc_name}", None)
                            delete_document(doc_name, user_id)
                            st.rerun()
                        if col_no.button("❌ Cancelar", key=f"no_{doc_name}", use_container_width=True):
                            st.session_state.pop(f"confirm_del_{doc_name}", None)
                            st.rerun()
        else:
            st.info("No hay documentos indexados aún.")

        # Clear pending reports now that they've been rendered
        if reports:
            st.session_state.pending_reports = []

        st.markdown("---")
        st.caption(f"Session: `{user_id[:8]}...`")

    # -----------------------------------------------------------------------
    # Main area: full-width chat
    # -----------------------------------------------------------------------
    st.title("💬 Chat con tus documentos")
    st.caption(
        "Sube tus documentos desde el panel izquierdo (PDF, Word, Excel, CSV o texto) "
        "y hazle preguntas en lenguaje natural. El agente buscará automáticamente "
        "la información más relevante en tu contenido y te dará una respuesta precisa, "
        "sin acceso a internet y sin salirse de lo que hay en tus archivos."
    )
    st.markdown("---")

    messages = SessionManager.get_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu pregunta aquí...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        SessionManager.add_message("user", user_input)

        agent = load_agent(user_id)

        if agent is None:
            error_msg = (
                "El agente no está disponible. Verifica que `GROQ_API_KEY` "
                "esté configurada en tu archivo `.env`."
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
                        handle_agent_error(e)


def _parse_retry_time(error_text: str) -> str | None:
    """Extract the wait time from a Groq rate-limit error message."""
    match = re.search(r"Please try again in ([^\.]+)", error_text)
    return match.group(1).strip() if match else None


def handle_agent_error(e: Exception) -> None:
    """Render a user-friendly error message, with special handling for rate limits."""
    error_text = str(e)
    is_rate_limit = "rate_limit_exceeded" in error_text or "429" in error_text

    if is_rate_limit:
        wait_time = _parse_retry_time(error_text)
        wait_msg = f"Por favor, vuelve a intentarlo en **{wait_time}**." if wait_time else "Por favor, inténtalo más tarde."
        st.warning(
            "⚠️ **Límite de tokens alcanzado**\n\n"
            "Esta aplicación usa la API gratuita de Groq, que tiene un límite diario de tokens. "
            f"Has agotado el límite disponible por hoy.\n\n"
            f"{wait_msg}"
        )
    else:
        st.error(f"Error al procesar tu consulta: {e}")


if __name__ == "__main__":
    main()
