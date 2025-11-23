from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(title="FastAPI Docker Starter")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Hello from FastAPI in Docker!"}


@app.post("/text")
async def receive_text(content: str = Form(...)):
    """Recibe texto a través de un form field `content`."""
    if not content.strip():
        raise HTTPException(status_code=400, detail="Empty content")
    # Respuesta mínima
    return JSONResponse({"received": True, "type": "text", "length": len(content)})


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
