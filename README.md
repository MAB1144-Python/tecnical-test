# FastAPI + Docker - Proyecto inicial

Proyecto mínimo para empezar con FastAPI y Docker - incluye endpoints para subir imágenes y audios,
un helper RAG sobre archivos en `source/`, y utilidades para transcribir audio y generar respuestas habladas.

Estructura principal
- `app/` - aplicación FastAPI
	- `main.py` - rutas y lógica principal
	- `whisper.py` - utilidades para transcripción (Whisper/local o API)
	- `rag_faq.py` - funciones RAG sobre la carpeta `source/`
	- `text_to_voice.py` - genera MP3 desde texto
	- `text_into_image.py` - extrae texto desde imágenes usando la Responses API (visión)
	- `input_audio/`, `input_images/`, `input_text/` - directorios donde se guardan uploads
	- `static/` - archivos estáticos servidos (respuestas habladas)
- `source/` - documentos de referencia (FAQ, manuales, errores comunes)
- `Dockerfile` - construcción de la imagen
- `docker-compose.yml` - levantar servicio en desarrollo
- `requirements.txt` - dependencias Python

Configuración (.env)
---------------------
Este proyecto usa la API de OpenAI para embeddings/LLM y/o la API de transcripción si no tienes Whisper local.
Crea un archivo `.env` en la raíz del proyecto con al menos la variable:

```
OPENAI_API_KEY=sk-...tu_clave_aqui...
```

Opcionalmente puedes añadir:

```
# Variables adicionales (ejemplos)
LANGUAGE_DEFAULT=es
```

IMPORTANTE: No incluyas claves reales en repositorios públicos; usa secretos de CI/CD o vaults en producción.

Construir y ejecutar
--------------------
1) Construir la imagen (desde la raíz del proyecto):

```bash
docker compose build --no-cache
```

2) Levantar en modo desarrollo (uvicorn con reload):

```bash
docker compose up
```

3) Abrir la API en: http://localhost:8000

Rutas principales y ejemplos
---------------------------
- POST /support
	- Form-data: `text` (string), `image` (file)
	- Ejemplo:
		```bash
		curl -X POST "http://localhost:8000/support" -F "text=Mi pantalla se queda en negro" -F "image=@/ruta/captura.png"
		```
	- Respuesta (JSON):
		```json
		{
			"transcription": "Texto extraído de la imagen",
			"answer": "Respuesta del asistente basada en RAG",
			"audio_url": "http://localhost:8000/static/respuesta_2025...mp3",
			"source_documents": ["FAQ_General.txt", "Errores Comunes en SoftHelp.txt"]
		}
		```

- POST /support/audio
	- Form-data: `audio` (file .mp3/.wav), `image` (file)
	- Ejemplo:
		```bash
		curl -X POST "http://localhost:8000/support/audio" -F "audio=@/ruta/audio.mp3" -F "image=@/ruta/captura.png"
		```
	- Respuesta (JSON): similar a /support, incluye `transcription` del audio y `extracted_text` de la imagen.

Notas de desarrollo
-------------------
- Si deseas usar la transcripción local con Whisper, instala `openai-whisper` y asegúrate de que `trascription_audio` funcione localmente.
- El módulo `rag_faq.py` carga documentos desde `source/` y construye un índice FAISS en memoria para responder consultas.
- Archivos subidos se guardan en `app/input_audio` y `app/input_images`.

Pruebas
------
Se pueden añadir pruebas con pytest y `httpx.AsyncClient` apuntando a la app FastAPI; no hay tests incluidos por defecto en este scaffold.

Soporte
-------
Para preguntas, abre un issue o contacta al autor del repositorio.

