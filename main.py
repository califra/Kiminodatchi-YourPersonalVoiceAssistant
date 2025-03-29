import builtins
import io
import os
import time

import google.generativeai as genai
import noisereduce as nr  # Optional noise reduction
import numpy as np
import pyaudio
import whisper
from dotenv import load_dotenv
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from pydub.playback import play

RATE = 16000  # Whisper expects 16kHz
CHUNK = 4096
SILENCE_DURATION = 2  # Stop transcription after 3 seconds of silence
MODEL_SIZE = "base"
SILENCE_THRESHOLD = 0.01

def is_silent(audio_data, threshold=SILENCE_THRESHOLD):
    """Check if the audio is silent based on RMS energy."""
    energy = np.sqrt(np.mean(np.square(audio_data)))
    return energy < threshold


# Load Whisper model for speech-to-text
print(f"Loading Whisper model: {MODEL_SIZE}...")
stt_model = whisper.load_model(MODEL_SIZE)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load the LLM for the chat
LLM = genai.GenerativeModel(model_name="gemini-1.5-flash")

def live_transcription(sample_rate=RATE, chunk_size=CHUNK):
    f"""
    Performs real-time speech-to-text transcription using OpenAI's Whisper model.
    Stops when there is no speech detected for {SILENCE_DURATION} consecutive seconds.
    """

    # Initialize PyAudio stream for real-time audio capture
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )

    print(
        f"🎤 Listening... Speak now! (Stops when you are silent for {SILENCE_DURATION} seconds)"
    )

    buffer = []
    last_speech_time = time.time()

    try:
        transcription_text = ""
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

        result = stt_model.transcribe(audio_segment, verbose=False)

        transcription_text = result["text"].strip()

        return transcription_text

    except KeyboardInterrupt:
        transcription_text = "INTERRUPT"
    finally:
        # Clean up resources
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()
    return transcription_text

def google_translate_tts(text):
    """Automatically detects language and plays speech directly from memory without saving."""
    if not text.strip():
        return

    try:
        lang = (
            detect(text) if len(text.split(" ")) > 2 else "en"
        )  # Auto-detect the language
        print(f"🔍 Detected language: {lang}")

        # Generate speech in memory (no file saving)
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)  # Move to the start of the buffer

        # Convert MP3 to WAV in-memory and play
        audio = AudioSegment.from_file(audio_fp, format="mp3")
        play(audio)
    except Exception as e:
        print(f"❌ Error in TTS: {e}")


chat = LLM.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are a helpful AI. Keep responses concise but not too short but avoid unnecessary explanations or questions."
            ],
        }
    ]
)

while True:
    try:
        transcription_text = live_transcription()
        # Ensure that the transcription text is not empty before sending it to the LLM
        if transcription_text != "" and transcription_text != "INTERRUPT":
            print(f"User: {transcription_text}")
            llm_response = chat.send_message(transcription_text).text
            google_translate_tts(llm_response)
            print("AI:", llm_response)
        else:
            if transcription_text == "INTERRUPT":
                break
    except KeyboardInterrupt:
        transcription_text = None
        break
