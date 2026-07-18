# Dockerfile for the rag-pipeline app service.
# Build context is ./app (see docker-compose.yml), so this copies the app package
# to /code/app and runs the API from /code with `app.main:app`.
#
# NOTE: This image is kept correct but was NOT built in the dev environment
# (no Docker available there). It should build and run as written.

FROM python:3.11-slim

WORKDIR /code

# Install the default (offline) dependencies.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source into the `app` package.
COPY . ./app

EXPOSE 8000

# Serve the FastAPI app. The in-memory store + offline LLM fallback mean this
# runs with no API key; set ANTHROPIC_API_KEY to enable generated answers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
