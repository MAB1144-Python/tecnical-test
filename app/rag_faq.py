"""RAG helper para múltiples fuentes usando LangChain.

Funciones principales:
- answer_question(...): responde usando RAG con todas las fuentes en la carpeta `source`.

Las fuentes deben estar en: <project_root>/source
y pueden ser archivos .txt, .md, .log (texto plano).
"""

from typing import Optional, List
from pathlib import Path
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Cargar .env si está disponible (no falla si python-dotenv no está instalado)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _load_source_documents(source_dir: Path) -> List[Document]:
    """
    Lee todos los documentos de source_dir y retorna una lista de Document.

    ✔ TXT, MD, LOG → lectura directa
    ✔ PDF → PyPDFLoader → documentos por página
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"La carpeta de fuentes no existe: {source_dir}")

    text_exts = {".txt", ".md", ".log", ".text"}

    docs: List[Document] = []

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()

        # === PDF ===
        if ext == ".pdf":
            try:
                loader = PyPDFLoader(str(path))
                pdf_docs = loader.load()
                for d in pdf_docs:
                    d.metadata["source"] = str(path)
                docs.extend(pdf_docs)
            except Exception as e:
                print(f"[WARN] No se pudo procesar PDF {path}: {e}")

        # === TEXTOS simples ===
        elif ext in text_exts:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(path)},
                        )
                    )
            except Exception as e:
                print(f"[WARN] No se pudo leer {path}: {e}")

        # Otros tipos se ignoran
        else:
            continue

    if not docs:
        raise ValueError(f"No se encontraron documentos válidos en {source_dir}")

    print(f"[INFO] Documentos cargados: {len(docs)}")
    return docs

def answer_question(
    query: str,
    message: str,
    top_k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
):
    """Responde una pregunta usando RAG con múltiples fuentes en la carpeta `source`.

    Parámetros:
        query: Pregunta o intención del usuario.
        message: Mensaje de sistema / texto de error que ve el usuario.
        top_k: Número de chunks a recuperar desde el vector store.
        embedding_model: Modelo de embeddings de OpenAI a usar.
        llm_model: Modelo de chat de OpenAI para generar la respuesta.
        api_key: (opcional) API key de OpenAI. Si se pasa, se usa en vez de la env var.
    """

    # Establecer API Key si se pasa explícitamente
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY no está definido. "
            "Establece la variable de entorno o pásala como api_key al llamar a answer_question."
        )

    # Raíz del proyecto (dos niveles arriba de este archivo, ajusta si es necesario)
    project_root = Path(__file__).resolve().parents[1]

    # Carpeta donde están las fuentes
    source_dir = project_root / "source"
    print(f"[INFO] Usando carpeta de fuentes: {source_dir}")

    # 1) Cargar todos los documentos de la carpeta source
    docs = _load_source_documents(source_dir)
    print(f"[INFO] Documentos cargados para RAG: {len(docs)}")
    # 2) Crear splitter y generar chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "P:", "R:", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    # 3) Embeddings y vector store
    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = FAISS.from_documents(split_docs, embeddings)

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # 4) LLM
    llm = ChatOpenAI(model=llm_model)

    # 5) Prompt RAG
    prompt = ChatPromptTemplate.from_template(
        """
## Role
Eres un asistente de soporte de la plataforma SoftHelp.

## Tasks
Analizar el **input** que está compuesto por un mensaje del usuario (**Message_User**)
y un texto de un mensaje que le aparece en pantalla (**Message_System**). Responder preguntas usando SOLO el contexto.

## Rules
- Responde en español de forma clara y breve usando EXCLUSIVAMENTE la información del contexto.
- Si la respuesta no está en el contexto, responde: "No encuentro esa información en las preguntas frecuentes."

## CONTEXTO
{context}

## PREGUNTA DEL USUARIO
{input}

## OUTPUT FORMAT
Responde solo con el texto de la respuesta en idioma frances, sin comillas ni formato adicional, en texto plano en idioma frances, respuesta debe entregarse en frances.
"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    input_data = f"""Message_User: {query}
Message_System: {message}
"""

    resp = rag_chain.invoke({"input": input_data})

    if isinstance(resp, dict):
        answer = resp.get("answer") or resp.get("output_text") or resp.get("result")
    elif isinstance(resp, str):
        answer = resp
    else:
        answer = str(resp)

    # Extract source documents from possible response fields
    source_set = set()
    def _add_source(src):
        if not src:
            return
        try:
            name = Path(str(src)).name
        except Exception:
            name = str(src)
        if name:
            source_set.add(name)

    if isinstance(resp, dict):
        # keys that may contain document references
        candidate_keys = ["context", "source_documents", "references", "docs", "metadata"]
        for key in candidate_keys:
            items = resp.get(key)
            if not items:
                continue
            # normalize to iterable
            if isinstance(items, dict):
                items_iter = [items]
            else:
                try:
                    items_iter = list(items)
                except Exception:
                    items_iter = [items]

            for it in items_iter:
                # Document object from LangChain
                try:
                    if hasattr(it, 'metadata') and isinstance(it.metadata, dict):
                        _add_source(it.metadata.get('source'))
                        continue
                except Exception:
                    pass

                # dict-like
                if isinstance(it, dict):
                    _add_source(it.get('source') or (it.get('metadata') or {}).get('source'))
                    continue

                # fallback: add string representation
                _add_source(str(it))

    source_docs = list(source_set)
    return {"answer": answer, "source_documents": source_docs}
