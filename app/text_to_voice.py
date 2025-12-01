from gtts import gTTS
from datetime import datetime
import os
from openai import OpenAI


def text_to_speech(text, lang='en'):
    """Convierte texto a voz y guarda el archivo de audio.

    Args:
        text: Texto a convertir.
        lang: Idioma del texto (por defecto 'en' para inglés).
        filename: Nombre del archivo de salida (por defecto 'output.mp3').

    Returns:
        Ruta del archivo de audio generado.
    """
    # Generate filename with timestamp: YYYYMMDD_HHMMSS_output.mp3
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"respuesta.mp3"

    # Ensure output directory exists inside app
    out_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, filename)
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path


def text_to_speech_openai_stream(text, voice='alloy', model='gpt-4o-mini-tts'):
    """Convierte texto a voz usando OpenAI API con streaming.

    Args:
        text: Texto a convertir.
        voice: Voz a usar (alloy, echo, fable, onyx, nova, shimmer).
        model: Modelo de OpenAI a usar (por defecto 'gpt-4o-mini-tts').

    Returns:
        Ruta del archivo de audio generado.
    """
    filename = f"respuesta_2.mp3"

    # Ensure output directory exists inside app
    out_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, filename)

    # Initialize OpenAI client
    client = OpenAI()
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(out_path)

    return out_path


