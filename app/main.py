from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from app.rag_faq import answer_question
from app.text_to_voice import text_to_speech

# Cargar variables de entorno desde .env si existe
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="FastAPI Docker Starter")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI in Docker!"}


@app.post("/ask")
async def ask(payload: dict = Body(...)):
    """Recibe JSON con {"question": "..."} y responde usando el RAG sobre el FAQ.

    Opciones de body:
    - {"question": "..."}
    - también se puede pasar "faq_path" o "persist_dir" para controlar el índice.
    """
    question = payload.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body")

    persist_dir = payload.get("persist_dir")

    res = answer_question(question, persist_dir=persist_dir)
    voice = text_to_speech(res["answer"], lang='es')
    res["audio_file"] = voice

    return JSONResponse(res)

@app.post("/image")
async def receive_image(file: UploadFile = File(...)):
    """Recibe una imagen como archivo. Se valida content-type básico."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    contents = await file.read()
    return {"received": True, "filename": file.filename, "content_type": file.content_type, "size": len(contents)}


@app.post("/audio")
async def receive_audio(file: UploadFile = File(...)):
    """Recibe un audio en mp3. Valida que sea audio/mpeg o .mp3."""
    ct = file.content_type
    filename = file.filename or ""
    if ct not in ("audio/mpeg", "audio/mp3") and not filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="File is not MP3 audio")
    contents = await file.read()
    return {"received": True, "filename": file.filename, "content_type": ct, "size": len(contents)}



