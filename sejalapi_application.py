from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import BartTokenizer, BartForConditionalGeneration
import whisper
import torch
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Function to transcribe audio file to text using Whisper model
def audio_to_text(audio_path: str) -> str:
    whisper_model = whisper.load_model("large-v3")
    audio_data = whisper.load_audio(audio_path)
    audio_data = whisper.pad_or_trim(audio_data)
    mel_spectrogram = whisper.log_mel_spectrogram(audio_data).to(whisper_model.device)
    decode_options = whisper.DecodingOptions(fp16=False)
    decode_result = whisper.decode(whisper_model, mel_spectrogram, decode_options)
    return decode_result.text

# Function to summarize text using BART model
def generate_summary(input_text: str) -> str:
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenized_input = bart_tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = bart_model.generate(tokenized_input['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary_text = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# Function to extract timestamps from the transcript
def get_timestamps(transcription: str) -> list:
    words_list = transcription.split()
    timestamps_list = [(i*5, (i+1)*5, ' '.join(words_list[i*50:(i+1)*50])) for i in range(len(words_list) // 50)]
    return timestamps_list

# Initialize FastAPI application
app = FastAPI()

# API endpoint to process audio files
@app.post("/process-audio/")
async def process_uploaded_audio(uploaded_file: UploadFile = File(...)):
    file_path = f"temp_{uploaded_file.filename}"
    with open(file_path, "wb+") as temp_file:
        temp_file.write(uploaded_file.file.read())
    
    try:
        # Transcribe the audio file to text
        transcription = audio_to_text(file_path)
        
        # Generate summary of the transcribed text
        summary_text = generate_summary(transcription)
        
        # Extract timestamps from the transcription
        transcription_timestamps = get_timestamps(transcription)
    except Exception as error:
        os.remove(file_path)
        return JSONResponse(content={"error": str(error)}, status_code=500)
    
    os.remove(file_path)
    return JSONResponse(content={
        "transcription": transcription,
        "summary": summary_text,
        "timestamps": transcription_timestamps
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
