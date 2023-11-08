from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from typing import List
import yt_dlp
import torch
import io
import os

app = FastAPI()

# Ensure the downloads directory exists
download_dir = 'downloads'
os.makedirs(download_dir, exist_ok=True)

# Function to set up the model and processor
def setup_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

# Initialize the model and processor
pipe = setup_model()

# Function to convert the sample rate of the audio file
def convert_sample_rate(audio_bytes):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    # Check if the sample rate is 16000 Hz, if not convert it
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    return audio

# Recursive function to get all mp3 files from a directory and its subdirectories
def get_mp3_files(directory: str) -> List[str]:
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    return mp3_files

# Function to transcribe audio using Whisper
def transcribe_audio(pipe, audio_bytes):
    # Convert the audio bytes to the correct sample rate
    audio = convert_sample_rate(audio_bytes)
    # Save the converted audio file to a temporary file
    temp_file = "temp_audio.mp3"
    audio.export(temp_file, format="mp3")

    # Run the transcription pipeline
    result = pipe(temp_file)
    return result["text"]

# Define the endpoint for audio file upload and transcription
@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        # Read the audio file
        audio_bytes = await file.read()
        # Transcribe the audio file
        transcription = transcribe_audio(pipe, audio_bytes)
        # Return the transcription
        return JSONResponse(content={"transcription": transcription}, status_code=200)
    except Exception as e:
        # Return the error
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.get("/transcribe_batch/")
async def transcribe_batch():
    try:
        batch_directory = 'batch'
        all_mp3_files = get_mp3_files(batch_directory)
        transcriptions = {}

        for audio_file_path in all_mp3_files:
            with open(audio_file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            transcription = transcribe_audio(pipe, audio_bytes)
            transcriptions[audio_file_path] = transcription

        return transcriptions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



######################################
#### Youtube to MP3 Converter API ####
######################################

# Function to download YouTube video and convert it to MP3
def youtube_video_to_mp3(youtube_url: str) -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'quiet': True,
        'no_warnings': True,
        'outtmpl': f'{download_dir}/%(title)s.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        video_title = info_dict.get('title', 'YouTubeAudio')
        ydl.download([youtube_url])

    # The expected filename of the audio file
    audio_filename = f'{download_dir}/{video_title}.mp3'
    
    return audio_filename

# Helper function to clean up the file after sending the response
def clean_up_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")

# Define the endpoint for YouTube to MP3 conversion
@app.post("/youtube_to_mp3/")
async def youtube_to_mp3(url: str, background_tasks: BackgroundTasks):
    try:
        audio_filename = youtube_video_to_mp3(url)

        # Create a FileResponse that prompts download on the client's side
        response = FileResponse(path=audio_filename, media_type='audio/mpeg', filename=os.path.basename(audio_filename))

        # Schedule the file to be deleted after the download
        background_tasks.add_task(clean_up_file, file_path=audio_filename)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
