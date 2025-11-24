from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from app.rag_faq import answer_question
from app.text_to_voice import text_to_speech
from app.whisper import transcribe_audio
from app.text_into_image import extraer_texto_error
import os
from datetime import datetime


# Cargar variables de entorno desde .env si existe
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="FastAPI Docker Starter")

@app.post("/support")
async def receive_image(text: str = Form(...), image: UploadFile = File(...)):
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
        extracted_text = extraer_texto_error(i_path)
    except Exception as e:
        extracted_text = f"image_extraction_error: {str(e)}"

    res = answer_question(question, message=extracted_text)
    voice = text_to_speech(res["answer"], lang='es')
    res["audio_file"] = voice
    
    return {
        "received": True,
        "image": {"saved_as": i_name, "saved_path": i_path, "size": len(image_bytes)},
        "extracted_text": extracted_text,
        "response": res
    }

@app.post("/support/audio")
async def receive_audio(audio: UploadFile = File(...), image: UploadFile = File(...)):
    """Recibe un audio (mp3/wav) y una imagen; valida tipos, guarda ambos y devuelve rutas y transcripción."""

    # --- Validate audio ---
    act = audio.content_type
    a_filename = audio.filename or ""
    allowed_audio_ct = ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave")
    if act not in allowed_audio_ct and not (a_filename.lower().endswith(".mp3") or a_filename.lower().endswith(".wav")):
        raise HTTPException(status_code=400, detail="Audio must be MP3 or WAV")

    audio_bytes = await audio.read()

    # --- Validate image ---
    ict = image.content_type
    i_filename = image.filename or ""
    if not ict or not ict.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image must be an image file")

    image_bytes = await image.read()

    # Ensure output directories exist
    base_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(base_dir, 'input_audio')
    images_dir = os.path.join(base_dir, 'input_images')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Build filenames
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    a_ext = ".mp3" if a_filename.lower().endswith(".mp3") or act.startswith("audio/mpeg") else ".wav"
    i_ext = os.path.splitext(i_filename)[1] or '.jpg'

    a_name = f"audio_input_{ts}{a_ext}"
    i_name = f"image_input_{ts}{i_ext}"

    a_path = os.path.join(audio_dir, a_name)
    i_path = os.path.join(images_dir, i_name)

    # Write to disk
    with open(a_path, 'wb') as f:
        f.write(audio_bytes)
    with open(i_path, 'wb') as f:
        f.write(image_bytes)

    # Transcribe audio (may raise if no backend)
    try:
        extracted_text = extraer_texto_error(i_path)
    except Exception as e:
        extracted_text = f"image_extraction_error: {str(e)}"
    
    try:
        transcription = transcribe_audio(a_path)
    except Exception as e:
        transcription = f"transcription_error: {str(e)}"

    return {
        "received": True,
        "audio": {"saved_as": a_name, "saved_path": a_path, "size": len(audio_bytes)},
        "image": {"saved_as": i_name, "saved_path": i_path, "size": len(image_bytes)},
        "transcription": transcription,
        "extracted_text": extracted_text
    }

