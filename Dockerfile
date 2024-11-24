# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (including git)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git curl && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Configure Git (optional, remove this if not required in production)
RUN git config --global http.postBuffer 524288000 && \
  git config --global http.sslVerify false

# Copy only requirements.txt first to leverage Docker cache for dependencies
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy the ./models directory into the image
COPY ./models /app/models

# Copy the ./data directory into the image
COPY ./data /app/data


# Expose the port your Flask app will run on
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Default command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
