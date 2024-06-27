# Audio Transcription and Summarization with FastAPI
This project is a FastAPI-based web service that allows users to upload audio files, which are then transcribed into text using the Whisper model. The transcribed text is summarized using a BART model, and timestamps are generated for segments of the transcription. The service returns the transcription, summary, and timestamps in a JSON response.
## Requirements
Install the required libraries:
pip install fastapi uvicorn whisper transformers torch python-dotenv
