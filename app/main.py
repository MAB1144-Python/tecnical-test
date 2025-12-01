from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from app.rag_faq import answer_question
from app.text_to_voice import text_to_speech, text_to_speech_openai_stream
from app.whisper import trascription_audio
from app.text_into_image import extraction_text
from fastapi.staticfiles import StaticFiles
import os
from datetime import datetime
import shutil

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="FastAPI Docker Starter")

# Mount static directory to serve response audio
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.post("/support")
async def receive_image_and_text(text: str = Form(...), image: UploadFile = File(...), request: Request = None):
    question = text
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'text' form field")
    
    # --- Validate image ---
    ict = image.content_type
    i_filename = image.filename or ""
    if not ict or not ict.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image must be an image file")

    image_bytes = await image.read()
    
    # Ensure output directories exist
    base_dir = os.path.dirname(__file__)
    images_dir = os.path.join(base_dir, 'input_images')
    os.makedirs(images_dir, exist_ok=True)
    i_ext = os.path.splitext(i_filename)[1] or '.jpg'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    i_name = f"image_input_{ts}{i_ext}"
    i_path = os.path.join(images_dir, i_name)
    with open(i_path, 'wb') as f:
        f.write(image_bytes)

    # Transcribe audio (may raise if no backend)
    try:
        extracted_text = extraction_text(i_path)
    except Exception as e:
        extracted_text = f"image_extraction_error: {str(e)}"

    res = answer_question(question, message=extracted_text)

    voice_path = text_to_speech(res.get("answer", ""), lang="es")
    static_name = "respuesta.mp3"
    audio_url = request.url_for("static", path=static_name)
    # Opcional: asegurar string
    audio_url = str(audio_url)

    transcription = "Don´t have audio transcription in this endpoint"
    return {
        "transcription": transcription,
        "extracted_text_of_image": extracted_text,
        "answer": res.get("answer"),
        "audio_url": audio_url,
        "source_documents": res.get("source_documents", []),
    }


@app.post("/support/audio")
async def receive_image_and_audio(audio: UploadFile = File(...), image: UploadFile = File(...), request: Request = None):
    # --- Validate audio ---
    act = audio.content_type
    a_filename = audio.filename or ""
    allowed_audio_ct = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave")
    if act not in allowed_audio_ct and not (a_filename.lower().endswith(".mp3") or a_filename.lower().endswith(".wav")):
        raise HTTPException(status_code=400, detail="Audio must be MP3 or WAV")

    audio_bytes = await audio.read()
    
    # Ensure output directories exist
    base_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(base_dir, 'input_audio')
    os.makedirs(audio_dir, exist_ok=True)

    # Build filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    a_ext = ".mp3" if a_filename.lower().endswith(".mp3") or act.startswith("audio/mpeg") else ".wav"
    a_name = f"audio_input_{ts}{a_ext}"
    
    a_path = os.path.join(audio_dir, a_name)
    # Write to disk
    with open(a_path, 'wb') as f:
        f.write(audio_bytes)
    
    # Transcribe audio
    try:
        transcription = trascription_audio(a_path)
    except Exception as e:
        transcription = f"transcription_error: {str(e)}"
        
    question = transcription
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'text' form field")
    
    # --- Validate image ---
    ict = image.content_type
    i_filename = image.filename or ""
    if not ict or not ict.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image must be an image file")

    image_bytes = await image.read()
    
    # Ensure output directories exist
    base_dir = os.path.dirname(__file__)
    images_dir = os.path.join(base_dir, 'input_images')
    os.makedirs(images_dir, exist_ok=True)
    i_ext = os.path.splitext(i_filename)[1] or '.jpg'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    i_name = f"image_input_{ts}{i_ext}"
    i_path = os.path.join(images_dir, i_name)
    with open(i_path, 'wb') as f:
        f.write(image_bytes)

    # Transcribe audio (may raise if no backend)
    try:
        extracted_text = extraction_text(i_path)
    except Exception as e:
        extracted_text = f"image_extraction_error: {str(e)}"

    res = answer_question(question, message=extracted_text)
    voice_path = text_to_speech(res.get("answer", ""), lang="es")
    static_name = "respuesta.mp3"
    audio_url = request.url_for("static", path=static_name)
    audio_url = str(audio_url)


    return {
        "transcription": transcription,
        "extracted_text_of_image": extracted_text,
        "answer": res.get("answer"),
        "audio_url": audio_url,
        "source_documents": res.get("source_documents", []),
    }
    
    
@app.post("/text_to_voice")
async def receive_image_and_text(text: str = Form(...), language: str = Form("es"), request: Request = None):
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' form field")

    if language not in ["es", "en"]:
        raise HTTPException(status_code=400, detail="Language must be 'es' or 'en'")
    voice_path = text_to_speech(text, lang=language)
    # Voice	Descripción
    # alloy	Voz estándar, clara y neutral.
    # verse	Voz cálida, más natural.
    # sage	Seria y profesional.
    # sol	Más dinámica y expresiva.
    # shimmer	Suave, femenina y cálida.
    # character	Más caricaturesca / expresiva.
    # Stream audio to file
    voice_path_2 = text_to_speech_openai_stream(text, voice='verse', model='gpt-4o-mini-tts')
    static_name = "respuesta.mp3"
    audio_url = request.url_for("static", path=static_name)
    # Opcional: asegurar string
    audio_url = str(audio_url)
    static_name_2 = "respuesta_2.mp3"
    audio_url_2 = request.url_for("static", path=static_name_2)
    # Opcional: asegurar string
    audio_url_2 = str(audio_url_2)
    return {
        "audio_url": audio_url,
        "audio_url_2": audio_url_2
    }
