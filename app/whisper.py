import os
from openai import OpenAI

def transcribe_audio(path: str) -> str:
    client = OpenAI()
    with open(path, "rb") as audio_file:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="es"
        )
    return resp.text
