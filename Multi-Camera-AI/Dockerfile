FROM python:3.10-slim

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install pip requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p database

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command: run FastAPI backend (in background) and Streamlit dashboard
CMD uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
