FROM python:3.11-slim

LABEL maintainer="your-team@example.com"
LABEL service="connect4-ml-api"
LABEL version="1.0.0"

WORKDIR /app

# Install minimal dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Model directory (mounted as volume)
RUN mkdir -p /app/models
VOLUME /app/models

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]