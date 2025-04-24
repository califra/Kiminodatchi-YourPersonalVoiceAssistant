FROM python:3.10-bullseye
ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg and system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    wget \
    ca-certificates \
    xz-utils \
    gcc \
    git \
    curl \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
 && wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
 && tar -xf ffmpeg-release-amd64-static.tar.xz \
 && cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ \
 && cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ \
 && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
 && rm -rf ffmpeg-*-amd64-static* \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
