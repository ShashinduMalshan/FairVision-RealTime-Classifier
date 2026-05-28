FROM python:3.11-slim

# Install system graphics dependencies that OpenCV requires to process video frames
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all your project files into the container space
COPY . .

# Install your Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default network communication port
EXPOSE 8501

# Run the application through the official Streamlit server command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]