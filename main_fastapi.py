import pdb
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import asyncio

# from asyncio import CancelledError
from contextlib import asynccontextmanager

import builtins
import io
import os
import time

from google import genai
from google.genai import types
import noisereduce as nr  # Optional noise reduction
import numpy as np
import pyaudio
import whisper
from dotenv import load_dotenv
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from pydub.playback import play
import tempfile

SYSTEM_NAME = "Kiminodatchi"
USERNAME = "You"

# --- Audio settings ---
RATE = 16000
CHUNK = 4096
SILENCE_DURATION = 3  # Stop transcription after 3 seconds of silence
MODEL_SIZE = "base"
SILENCE_THRESHOLD = 0.01
DEFAULT_LANGUANGE = "en"
SUPPORTED_LANGUAGES = ["en", "it", "fr", "ja", "zh-cn"]
# --- Global state ---
chat_queue = asyncio.Queue()
running = False
shutdown_event = asyncio.Event()


# --- Lifespan for graceful shutdown with FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI supports a lifespan context to run startup and shutdown code.
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    We use this to gracefully stop background loops and clean up."""

    print("FastAPI starting up...")
    yield
    print("FastAPI shutting down...")
    global running
    running = False
    shutdown_event.set()
    await chat_queue.put("Server is shutting down.")


app = FastAPI(lifespan=lifespan)

# --- Load models and API keys ---
print(f"Loading Whisper model: {MODEL_SIZE}...")
stt_model = whisper.load_model(MODEL_SIZE)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

model_instructions = """You are a helpful AI. 
    Keep responses concise but not too short.
    Avoid unnecessary explanations or questions."""
chat_config = types.GenerateContentConfig(system_instruction=model_instructions)
LLM_chat = client.chats.create(model="gemini-2.0-flash", config=chat_config, history=[])


# --- Utility Functions ---
def is_silent(audio_data, threshold=SILENCE_THRESHOLD):
    energy = np.sqrt(np.mean(np.square(audio_data)))
    return energy < threshold


def live_transcription(sample_rate=RATE, chunk_size=CHUNK):
    f"""
    Perform real-time speech-to-text transcription using OpenAI's Whisper model.
    Audio recording stops when there is no speech detected for {SILENCE_DURATION} consecutive seconds.
    """
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print(f"I am listening... Speak now! (Silence for {SILENCE_DURATION}s to stop)")

    buffer = []
    last_speech_time = time.time()

    try:
        while True:
            # Capture audio in real-time
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_data = (
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            )  # Normalize to [-1,1]
            audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate)
            buffer.extend(audio_data)
            # Check for silence
            if is_silent(audio_data):
                if time.time() - last_speech_time > SILENCE_DURATION:
                    break
            else:
                last_speech_time = (
                    time.time()
                )  # Reset silence timer when speech is detected

        # Transcribe the entire recorded audio
        audio_segment = np.array(buffer)
        transcription_text = stt_model.transcribe(audio_segment, verbose=False)[
            "text"
        ].strip()
        return transcription_text

    except KeyboardInterrupt:
        return "INTERRUPT"
    finally:
        # Clean up resources
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()


import wave


def save_temp_wav(audio_bytes: bytes, sample_rate: int = 16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        with wave.open(tmp_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return tmp_file.name


def live_transcription_from_bytes(audio_bytes: bytes):
    temp_path = save_temp_wav(audio_bytes)
    result = stt_model.transcribe(temp_path)
    os.remove(temp_path)  # Clean up after use
    return result["text"].strip()


def tts_to_bytes(text, lang="en"):
    # Generate speech in memory (no file saving)
    tts = gTTS(text=text, lang=lang)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)  # Move to the start of the buffer
    return audio_fp


def speak_response(text):
    # Convert AI text response to Audio. First detects language, then speaks.
    if not text.strip():
        return
    try:
        # Auto-detect the language
        lang = detect(text)  # if len(text.split(" ")) > 2 else "en"
        if lang not in SUPPORTED_LANGUAGES:
            print(f"lang detected is {lang}, defaulting to {DEFAULT_LANGUANGE}.")
            lang = DEFAULT_LANGUANGE
        print(f"Detected language: {lang}")

        # Generate speech in memory (no file saving)
        audio_fp = tts_to_bytes(text, lang)

        # Convert MP3 to WAV in-memory and play
        audio = AudioSegment.from_file(audio_fp, format="mp3")
        play(audio)
    except Exception as e:
        print(f"Error in TTS: {e}")


# --- Core background async task: AI conversation loop ---
# Reasons to use async:
# 1) live_transcription is CPU + I/O heavy, so it’s run using await asyncio.to_thread(...) which doesn’t block the event loop.
# 2) chat_queue.put() is async-safe communication — needed for streaming output later.
async def speak_to_ai():
    global running
    try:
        while running and not shutdown_event.is_set():
            transcription_text = await asyncio.to_thread(live_transcription)

            # Ensure that the transcription text is not empty before sending it to the LLM
            if transcription_text and transcription_text != "INTERRUPT":
                print(f"{USERNAME}: {transcription_text}")
                llm_response = LLM_chat.send_message(transcription_text).text
                print(f"{SYSTEM_NAME}:", llm_response)
                await asyncio.to_thread(speak_response, llm_response)
                await chat_queue.put(f"{USERNAME}: {transcription_text}")
                await chat_queue.put(f"{SYSTEM_NAME}: {llm_response}")
            elif transcription_text == "INTERRUPT":
                await chat_queue.put("Session ended by user.")
                break

    finally:
        print("Speak_to_ai terminated.")
        running = False


# --- FastAPI Routes ---
# Adds the AI loop task to background without blocking the API (if it’s not already running)
@app.post("/speak")
async def start_speaking_loop(background_tasks: BackgroundTasks):
    global running
    if not running:
        running = True
        background_tasks.add_task(speak_to_ai)
        return {"status": "Started voice chat loop"}
    else:
        return {"status": "Already running"}


# Stream Output (Server-Sent Events)
# Why SSE (Server-Sent Events)?
# This lets the client (browser, app, etc.) receive real-time updates over HTTP.
# Each message is sent like: data: ...\n\n
# Compatible with JavaScript EventSource or tools like React.
@app.get("/stream")
async def stream_chat():
    async def event_generator():
        while True:
            message = await chat_queue.get()
            yield f"data: {message}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/stop")
async def stop_loop():
    global running
    running = False
    await chat_queue.put("Session stopped manually.")
    return {"status": "Stopped"}


# --- WebSocket Audio Endpoint ---
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        audio_buffer = bytearray()
        while True:
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)
            # Optional: signal from frontend to end
            if len(chunk) < 1024:
                break

        audio_bytes = bytes(audio_buffer)
        if not audio_bytes:
            await websocket.send_text("No audio received.")
            return

        transcription_text = live_transcription_from_bytes(audio_bytes)
        if not transcription_text.strip():
            await websocket.send_text("Sorry, I didn't catch that.")
            return

        print(f"{USERNAME}: {transcription_text}")

        await websocket.send_text(f"{USERNAME}: {transcription_text}")

        llm_response = LLM_chat.send_message(transcription_text).text
        print(f"{SYSTEM_NAME}: {llm_response}")
        audio_fp = tts_to_bytes(llm_response)  # .get_value()
        await websocket.send_bytes(audio_fp.read())
        await websocket.send_text(f"{SYSTEM_NAME}: {llm_response}")
        # await asyncio.to_thread(speak_response, llm_response)  # <-- Add back vocal response
    except WebSocketDisconnect:
        print("WebSocket disconnected")
