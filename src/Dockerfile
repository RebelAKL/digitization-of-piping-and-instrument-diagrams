FROM python:3.9-slim-bullseye

LABEL pipelineName="pnide-webapi"

# Use root for installs
USER root

# Install system packages required for OpenCV, PDF/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Add app user
RUN adduser --disabled-password --gecos '' pythonwebapi

WORKDIR /home/pythonwebapi

# Copy requirements first (better Docker caching)
COPY requirements.txt requirements.txt
COPY logger_config.py logger_config.py

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Install app dependencies
# Note: requirements.txt should REMOVE pyodbc and ADD:
#   cloud-sql-python-connector[pymysql]
RUN pip install -r requirements.txt \
    && pip install gunicorn flake8

# Copy source code
COPY app app
COPY init_app.py ./

# Flask app variable for gunicorn/uvicorn
ENV FLASK_APP init_app.py

# Ensure correct ownership
RUN chown -R pythonwebapi:pythonwebapi ./
USER pythonwebapi

# Cloud Run expects the app to listen on $PORT (defaults 8080)
ENV PORT 8080

EXPOSE 8080

# Gunicorn + Uvicorn worker
ENTRYPOINT ["gunicorn", "init_app:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "--timeout", "180"]
