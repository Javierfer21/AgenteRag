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

        # --- Step 4: upsert to Pinecone ---
        pinecone_manager.upsert_chunks(
            chunks=chunks,
            user_id=user_id,
            embedding_model=embedding_model,
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

def main():
    SessionManager.init_session()
    user_id = SessionManager.get_user_id()

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
            for doc_name, meta in documents.items():
                icon = EXTENSION_ICONS.get(meta.get("extension", ""), "📄")
                size_kb = meta.get("size_bytes", 0) / 1024
                report = reports_by_name.get(doc_name)
                with st.expander(f"{icon} {doc_name}", expanded=(doc_name in reports_by_name)):
                    st.caption(f"**Tamaño:** {size_kb:.1f} KB &nbsp;·&nbsp; **Tipo:** {meta.get('extension','?').upper()}")
                    st.caption(f"**Caracteres:** {meta.get('char_count', 0):,} &nbsp;·&nbsp; **Palabras:** {report['word_count']:,}" if report else f"**Caracteres:** {meta.get('char_count', 0):,}")
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
                        error_msg = f"Error al procesar tu consulta: {e}"
                        st.error(error_msg)
                        SessionManager.add_message("assistant", error_msg)


if __name__ == "__main__":
    main()
