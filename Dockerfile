# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY app.py /app/app.py
COPY best_yolov8m_model.pt /app/best_yolov8m_model.pt
COPY requirements.txt /app/requirements.txt

# Install required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port for the API
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
