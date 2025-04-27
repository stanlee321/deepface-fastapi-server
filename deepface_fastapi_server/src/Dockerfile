# Use an official Python runtime as a parent image
# Consider using a slim variant for smaller image size, e.g., python:3.9-slim
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed by libraries like OpenCV or Pillow
# Update package list and install essentials, then clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    # Add other system dependencies needed by your Python packages
    # For OpenCV, common dependencies include:
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
# Copy only the requirements file first to leverage Docker cache
COPY ./src/requirements.txt /app/src/requirements.txt

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/src/requirements.txt

# Copy the rest of the application code into the container
# Copying src content into /app/src within the container
COPY ./src /app/src

# Expose the port the app runs on (should match Uvicorn command in docker-compose)
EXPOSE 8000

# Define the command to run the application (matches docker-compose command)
# Note: Uvicorn command needs to reference the app correctly based on WORKDIR
# Since WORKDIR is /app, the path to main is src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 