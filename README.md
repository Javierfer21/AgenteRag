# 🤖 AgenteRag

**AgenteRag** es un agente conversacional inteligente basado en RAG (Retrieval Augmented Generation) que te permite chatear con tus documentos usando inteligencia artificial.

## 🚀 Tecnologías

| Componente | Tecnología |
|-----------|-----------|
| **Interfaz de Usuario** | Streamlit |
| **Agente AI** | LangGraph + LangChain |
| **Modelo de Lenguaje** | Groq API (llama-3.3-70b-versatile) |
| **Vector Store** | Pinecone |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Memoria** | SQLite |
| **Procesadores** | PyPDF2, python-docx, pandas, openpyxl |

## 📋 Requisitos Previos

- Python 3.11 o superior
- Cuenta en [Groq](https://console.groq.com/) (gratuita)
- Cuenta en [Pinecone](https://www.pinecone.io/) (plan gratuito disponible)

## 📦 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd c:\Users\njavi\Desktop\AgenteRag
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copia el archivo de ejemplo y completa tus claves:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edita el archivo `.env` con tus credenciales:

```env
# Groq API - Obtén en: https://console.groq.com/
GROQ_API_KEY=gsk_tu_clave_aqui

# Pinecone - Obtén en: https://www.pinecone.io/
PINECONE_API_KEY=tu_clave_pinecone_aqui
PINECONE_INDEX_NAME=agente-rag
PINECONE_ENVIRONMENT=us-east-1
```

## 🎯 Uso

### Iniciar la aplicación

```bash
# Desde el directorio del proyecto
streamlit run app/main.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

### Cómo usar AgenteRag

1. **Verificar configuración**: En el panel lateral verás el estado de las APIs (verde = OK)

2. **Subir documentos**:
   - Haz clic en "Selecciona archivos" en el sidebar
   - Sube PDF, DOCX, TXT, CSV o Excel
   - Espera a que se procese e indexe

3. **Hacer preguntas**:
   - Escribe tu pregunta en el cuadro de chat
   - El agente buscará en tus documentos y responderá

4. **Ejemplos de preguntas**:
   - "¿Qué documentos tengo disponibles?"
   - "Resume el informe de ventas"
   - "¿Cuál es el total de ventas del Q3?"
   - "¿Qué dice el contrato sobre penalizaciones?"
   - "Calcula el 15% de 45,000"

## 📁 Formatos Soportados

| Formato | Extensión | Descripción |
|---------|-----------|-------------|
| PDF | `.pdf` | Documentos PDF con texto extraíble |
| Word | `.docx` | Documentos Microsoft Word |
| Texto | `.txt`, `.md`, `.log` | Archivos de texto plano |
| CSV | `.csv`, `.tsv` | Datos separados por comas/tabuladores |
| Excel | `.xlsx`, `.xls`, `.xlsm` | Hojas de cálculo Excel |

> **Nota**: Los PDFs escaneados (imágenes) no son procesables sin OCR adicional.

## 🛠️ Herramientas del Agente

El agente tiene las siguientes herramientas disponibles:

| Herramienta | Descripción |
|-------------|-------------|
| `buscar_en_documentos` | Búsqueda semántica en documentos indexados |
| `resumir_documento` | Genera resumen de un documento específico |
| `analizar_datos_csv` | Análisis estadístico de datos CSV |
| `calculadora` | Cálculos matemáticos seguros |
| `listar_documentos` | Lista los documentos disponibles |

## 🏗️ Estructura del Proyecto

```
AgenteRag/
├── app/
│   ├── __init__.py
│   ├── config.py          # Configuración con pydantic-settings
│   ├── main.py            # Aplicación Streamlit principal
│   └── session_manager.py # Gestión de sesiones Streamlit
├── core/
│   ├── __init__.py
│   ├── agent.py           # Agente LangGraph principal
│   ├── memory.py          # Memoria conversacional SQLite
│   ├── state.py           # Estado del agente (TypedDict)
│   └── tools/
│       ├── __init__.py
│       ├── analyze_csv.py
│       ├── calculator.py
│       ├── list_documents.py
│       ├── search_documents.py
│       └── summarize_document.py
├── processors/
│   ├── __init__.py        # Fábrica de procesadores
│   ├── base.py            # Clase base abstracta
│   ├── csv_processor.py
│   ├── docx_processor.py
│   ├── excel_processor.py
│   ├── pdf_processor.py
│   └── txt_processor.py
├── vectorstore/
│   ├── __init__.py
│   ├── embeddings.py      # SentenceTransformers wrapper
│   └── pinecone_manager.py
├── utils/
│   ├── __init__.py
│   ├── chunking.py        # Chunking de texto
│   └── hash_utils.py      # Utilidades de hash MD5
├── tests/
│   ├── __init__.py
│   ├── test_calculator.py
│   ├── test_chunking.py
│   ├── test_hash_utils.py
│   ├── test_memory.py
│   └── test_processors.py
├── data/                  # Base de datos SQLite (auto-creada)
├── .env                   # Variables de entorno (crear desde .env.example)
├── .env.example
├── requirements.txt
└── README.md
```

## 🧪 Ejecutar Tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Test específico
python -m pytest tests/test_calculator.py -v

# Con coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

## ⚙️ Configuración Avanzada

### Variables de entorno completas

| Variable | Valor por defecto | Descripción |
|----------|------------------|-------------|
| `GROQ_API_KEY` | - | **Requerida** Clave API de Groq |
| `PINECONE_API_KEY` | - | **Requerida** Clave API de Pinecone |
| `PINECONE_INDEX_NAME` | `agente-rag` | Nombre del índice en Pinecone |
| `PINECONE_ENVIRONMENT` | `us-east-1` | Región de Pinecone |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Modelo de Groq a usar |
| `LLM_TEMPERATURE` | `0.1` | Temperatura del LLM (0.0-1.0) |
| `LLM_MAX_TOKENS` | `4096` | Máximo de tokens en respuesta |
| `CHUNK_SIZE` | `1000` | Tamaño de chunks en caracteres |
| `CHUNK_OVERLAP` | `200` | Solapamiento entre chunks |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modelo de embeddings |
| `EMBEDDING_DIMENSION` | `384` | Dimensión de embeddings |
| `SQLITE_DB_PATH` | `./data/memory.db` | Ruta a la base de datos |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `DEBUG` | `false` | Modo debug |

### Personalizar el tamaño de chunks

Para documentos técnicos largos, puedes aumentar el chunk size:

```env
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

Para documentos cortos o Q&A, reduce el chunk size:

```env
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

## 🔐 Seguridad

- **Nunca** incluyas el archivo `.env` en control de versiones
- El archivo `.gitignore` ya excluye `.env` y la carpeta `data/`
- Los IDs de usuario son UUID aleatorios generados por sesión
- La calculadora usa evaluación segura de AST (no `eval()`)
- Los documentos se separan por namespace en Pinecone

## 🐛 Solución de Problemas

### Error: "GROQ_API_KEY no configurada"
Verifica que el archivo `.env` existe y tiene la clave correcta.

### Error: "No se pudo conectar con Pinecone"
1. Verifica que `PINECONE_API_KEY` es válida
2. El primer uso puede tardar mientras se crea el índice
3. Verifica que `PINECONE_ENVIRONMENT` coincide con tu región

### El modelo no encuentra información en documentos
1. Verifica que el documento fue subido e indexado (panel lateral)
2. Reformula la pregunta con términos más específicos
3. Verifica que el PDF tiene texto extraíble (no es solo imágenes)

### Error de memoria o rendimiento
- Reduce `CHUNK_SIZE` para documentos grandes
- Los modelos de embedding se cargan una vez en memoria
- Usa `all-MiniLM-L6-v2` que es ligero pero efectivo

## 📊 Arquitectura del Agente

```
Usuario → Streamlit UI → RAGAgent
                              ↓
                    [LangGraph StateGraph]
                              ↓
                       nodo_agente (LLM)
                         ↙      ↘
              herramientas    respuesta directa
                    ↓
              ToolNode (ejecuta)
                    ↓
              nodo_agente (LLM)
                    ↓
              respuesta final → Usuario
```

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Escribe tests para tu código
4. Haz commit (`git commit -m 'Agrega nueva funcionalidad'`)
5. Haz push (`git push origin feature/nueva-funcionalidad`)
6. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

Desarrollado con ❤️ usando LangGraph, Groq y Pinecone.
