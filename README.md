# Audio Transcription and Summarization with FastAPI
This project is a FastAPI-based web service that allows users to upload audio files, which are then transcribed into text using the Whisper model. The transcribed text is summarized using a BART model, and timestamps are generated for segments of the transcription. The service returns the transcription, summary, and timestamps in a JSON response.
## Requirements
Install the required libraries:

`pip install fastapi uvicorn whisper transformers torch python-dotenv`

## Usage
Start the Server
Run the FastAPI server using:

`uvicorn api_application:app --reload`

## API Endpoint
POST `/process-audio/`
Upload an audio file to transcribe, summarize, and extract timestamps.

## Request:
*Method: POST
*Content-Type: multipart/form-data
*Form Data:
`file`: 
  *The audio file to be processed (.wav, .mp3, etc.)
Response:
*transcription: The full transcription of the audio.
*summary_text: A summarized version of the transcription.
*transcription_timestamps: Extracted timestamps with corresponding transcript segments.
Example
You can use `curl` to test the endpoint:

sh
Copy code
curl -X POST `"http://localhost:8000/process-audio/" -F "file=@/path/to/your/audiofile.mp3"`
## Project Structure
.
├── sejal.py          # Main FastAPI application
├── .env             # Environment variables (if needed)
├── README.md        # Project documentation
