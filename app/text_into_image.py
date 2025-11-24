from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extraer_texto_error(imagen_path: str) -> str:
    image_b64 = image_to_base64(imagen_path)

    prompt = (
        """
        ## Role
        Extrae TODO el texto que ves en esta captura de pantalla de un software. 
        ## Rules
        - Respóndelo tal cual aparece, sin comentar nada más. 
        - Incluye códigos de error, rutas de archivos y mensajes en inglés o español.
        """
    )

    response = client.responses.create(
        model="gpt-4.1-mini",   # o gpt-4.1, gpt-4.1-preview, etc. con visión
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ],
    )

    # El texto suele venir en response.output[0].content[0].text
    return response.output[0].content[0].text