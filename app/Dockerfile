# Dockerfile for the rag-pipeline app service.
# TODO: MEMORY.md checklist — "docker compose up runs the pipeline + vector DB + UI/API"
# NOTE: Scaffolding only. Do NOT `docker build` yet.

FROM python:3.11-slim

WORKDIR /code

# Install intended dependencies (build step is left for implementation phase).
COPY requirements.txt ./requirements.txt
# TODO: enable once dependencies are finalized:
# RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY . ./app

EXPOSE 8000

# TODO: run the API server once main.create_app() is implemented.
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "-c", "print('rag-pipeline scaffold: app not implemented yet')"]
