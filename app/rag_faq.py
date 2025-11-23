"""RAG helper para el FAQ usando LangChain.

Funciones principales:
- build_faq_index(faq_path, persist_dir=None): indexa el FAQ usando FAISS.
- answer_question(...): responde usando el índice (persistente o temporal).

Este módulo realiza imports dinámicos dentro de las funciones para que el módulo
pueda importarse si las dependencias no están instaladas; las funciones lanzarán
ImportError con mensajes claros si faltan paquetes.
"""
from typing import Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
import os

# Cargar .env si está disponible (no falla si python-dotenv no está instalado)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def answer_question(
    query: str,
    persist_dir: Optional[str] = None,
    top_k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
):
    """Responde una pregunta usando RAG.

    Si `persist_dir` contiene un índice lo carga; si no, requiere `faq_path`.
    """

    # usar ruta relativa desde la raíz del proyecto
    project_root = Path(__file__).resolve().parents[1]
    faq_path = project_root / "TechnicalInterview" / "Preguntas Frecuentes (FAQ).txt"

    # Permitir pasar la api_key directamente
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY no está definido. Establece la variable de entorno o pásala como api_key al llamar a answer_question."
        )

    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = None


    # construir en memoria (igual que build_faq_index)
    faq_path_obj = faq_path.resolve() if not faq_path.is_absolute() else faq_path
    print(faq_path_obj)
    # Debug: listar archivos en la carpeta `app` del proyecto en logs de arranque
    project_app_dir = Path(__file__).resolve().parents[1] / "app"
    
    if not faq_path_obj.exists():
        raise FileNotFoundError(f"FAQ not found************: {faq_path_obj}")

    faq_text = faq_path_obj.read_text(encoding="utf-8")
    docs = [Document(page_content=faq_text, metadata={"source": str(faq_path_obj)})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "P:", "R:", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    store = FAISS.from_documents(split_docs, embeddings)

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    # LLM
    llm = ChatOpenAI(model=llm_model)

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
## Role
Eres un asistente de soporte de la plataforma SoftHelp.

## Rules
- Responde en español de forma clara y breve usando EXCLUSIVAMENTE la información del contexto.
- Si la respuesta no está en el contexto, responde: "No encuentro esa información en las preguntas frecuentes."

## CONTEXTO
{context}

## PREGUNTA DEL USUARIO
{input}

## OUTPUT FORMAT
Responde solo con el texto de la respuesta, sin comillas ni formato adicional. en formato texto plano.

"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # Ejecutar chain
    try:
        resp = rag_chain.invoke({"input": query})
    except Exception:
        # algunos bindings usan run
        resp = {"answer": rag_chain.run(query)}
    # Normalizar respuesta
    answer = None
    if isinstance(resp, dict):
        answer = resp.get("answer") or resp.get("output_text") or resp.get("result")
    elif isinstance(resp, str):
        answer = resp
    else:
        answer = str(resp)

    return {"answer": answer}
