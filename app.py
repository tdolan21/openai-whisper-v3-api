import streamlit as st
import requests

st.title("Audio Transcription App")
st.write("Upload an audio file and it will be transcribed using the Whisper model through the FastAPI endpoint.")

# File uploader
audio_file = st.sidebar.file_uploader("Upload your audio file", type=['wav', 'mp3', 'ogg'])

if audio_file is not None:
    # Inform the user that the transcription is in process
    with st.spinner('Transcribing...'):
        # Prepare the file to send to the FastAPI endpoint
        files = {"file": (audio_file.name, audio_file, "audio/mpeg")}
        # POST the file to the FastAPI endpoint
        response = requests.post("http://localhost:8000/transcribe/", files=files)

        if response.status_code == 200:
            # Display the transcription
            transcription = response.json()["transcription"]
            st.text("Transcription:")
            st.write(transcription)
        else:
            # Handle errors
            st.error("An error occurred during the transcription process")
            st.write(response.json()["error"])

