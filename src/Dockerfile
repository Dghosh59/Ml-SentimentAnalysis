# Use an official Python runtime as base image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file first (for caching)
COPY src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY src/ .

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
