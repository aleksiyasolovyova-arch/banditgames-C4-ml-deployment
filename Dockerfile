FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
# We copy src/ so that 'from src.preprocessing' works exactly like in training
COPY src/ /app/src/
COPY main.py .

# Create the models directory (Volume mount point)
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8001

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]