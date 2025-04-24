# Kiminodatchi

**Kiminodatchi** is a lightweight, real-time voice assistant you can talk to.  
I built it as a personal learning project to explore:

- Audio streaming and real-time interaction  
- Speech recognition and noise reduction  
- LLM integration and text-to-speech pipelines  

Some parts of this project were assisted by ChatGPT to explore best practices and accelerate learning.

**About the name:** Kiminodatchi is the fusion of Japanese Kimino Tomodatchi i.e. your friend.

### 🔧 Tech Highlights

- Frontend voice interaction via WebSockets + Web Audio API  
- Whisper for speech transcription  
- Gemini API for generating responses  
- gTTS + PyDub for voice output  
- Optional noise reduction with `noisereduce`  
- FastAPI backend for audio streaming and processing  

This project helped me better understand the full-stack pipeline connecting audio input/output and LLMs via streaming APIs.

---

## 📦 System Dependencies
This document outlines the steps to get Kiminodatchi up and running.  Navigate to the bottom if you want to use a Dockerized version.

Otherwise:

Make sure the following are installed on your system:

- **Python 3.10**
- **FFmpeg** (used by Whisper, PyDub, and gTTS)
- **PortAudio** (used by PyAudio)

Install them on macOS using Homebrew:

```bash
brew install python@3.10
brew install ffmpeg
brew install portaudio
```

---

## 🐍 Setup (with Python 3.10)

Create a virtual environment and install the required Python packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

---

## 🔑 API Key

Create a `.env` file in the project root with the following content:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🖥️ Interact via Terminal

```bash
source .venv/bin/activate
python main.py
```

---

## 🌐 Interact via Web Browser

Start the FastAPI backend:

```bash
source .venv/bin/activate
uvicorn main_fastapi:app --host=127.0.0.1 --port=8000
```

Then open `index.html` in your browser (tested with Safari and Chrome).

---

## 🧪 Alternative: Use curl (Terminal Interaction)

1. Start the FastAPI server:

   ```bash
   uvicorn main_fastapi:app
   ```

2. In another terminal, initiate the chat:

   ```bash
   curl -X POST http://localhost:8000/speak
   ```

3. In another terminal, stream the chat:

   ```bash
   curl -N http://localhost:8000/stream
   ```

4. When done, stop the chat:

   ```bash
   curl -X POST http://localhost:8000/speak
   ```

---

## 🐳 Alternative: Run with Docker

This method allows you to run Kiminodatchi in a self-contained Docker environment, ensuring consistency across different systems. You have two options: building the image yourself or using a pre-built image.

**Option 1: Build the Docker Image (Recommended for Local Development)**

1.  Navigate to the project directory containing the `Dockerfile`.

2.  Build the Docker image: This command will create a Docker image named `kiminodatchi-app` based on the instructions in the `Dockerfile`.

    ```bash
    docker build -t kiminodatchi-app .
    ```

**Option 2: Use the Pre-built Docker Image (Quick Start)**

1.  Pull the pre-built image: If you prefer a faster setup, you can download a pre-built image from Docker Hub.

    ```bash
    docker pull fcaliva/kiminodatchi-app:v0
    ```

**Running the Docker Container:**

Once you have either built the image or pulled the pre-built one, you can launch the Kiminodatchi application in a Docker container:

   ```bash
   docker run --env-file .env -p 8000:8000 kiminodatchi-app
   ```

## 🚧 TODOs

- [x] Docker packaging  
- [ ] Multi-turn memory  
- [ ] Kubernetes deployment  
