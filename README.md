# FastAPI + Docker - Proyecto inicial

Proyecto mínimo para empezar con FastAPI y Docker.

Estructura:

- `app/` - aplicación FastAPI
- `Dockerfile` - imagen de la app
- `docker-compose.yml` - servicio para desarrollo
- `requirements.txt` - dependencias
- `tests/` - pruebas pytest

Ejecutar con docker compose:


Con Docker:

```bash
docker compose up --build
```

Nota: `docker-compose.yml` está configurado para ejecutar Uvicorn con `--reload`, por lo que los cambios en `./app` se recargarán automáticamente en modo desarrollo.

