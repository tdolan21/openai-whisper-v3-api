from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import io

app = FastAPI()

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

