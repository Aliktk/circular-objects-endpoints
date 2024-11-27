# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application code
COPY ./app ./app

# Create directories for uploads, ground truth, and logs
RUN mkdir -p /app/uploads /app/ground_truth /app/logs

# Expose port
EXPOSE 8000

# Set environment variable for log file path
ENV LOG_FILE=/app/logs/app.log

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
