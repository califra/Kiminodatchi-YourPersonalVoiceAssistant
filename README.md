# MyPersonalVoiceAssistant
This project implements a very basic voice assistant you can chat with

# Lanch main file
Create a file named .env which looks like

    GEMINI_API_KEY=$YOUR_GEMINI_API_KEY 
Navigate to the project folder

    python main.py


# Lanch main_fastapi file 
This version supports FastAPI
Create a file named .env which looks like

    GEMINI_API_KEY=$YOUR_GEMINI_API_KEY 
Navigate to the project folder

    uvicorn main_fastapi:app

From another terminal initiate the chat using

    curl -X POST http://localhost:8000/speak

From another terminal monitor the chat using 

    curl -N http://localhost:8000/stream

Terminate the chat using

    curl -X POST http://localhost:8000/speak
