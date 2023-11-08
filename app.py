import streamlit as st
import requests
import os

st.title("OpenAI Whisper-Large-v3")

# File uploader in the sidebar for direct audio file upload
audio_file = st.sidebar.file_uploader("Upload your audio file", type=['wav', 'mp3', 'ogg'])
st.sidebar.divider()

# Text input in the sidebar for YouTube URL
youtube_url = st.sidebar.text_input("Or enter a YouTube URL here:")


# Button to trigger YouTube to MP3 conversion
if st.sidebar.button('Create MP3 from YouTube'):
    if youtube_url:
        with st.spinner('Downloading and converting YouTube video to MP3...'):
            response = requests.post("http://localhost:8000/youtube_to_mp3/", params={"url": youtube_url})
            if response.status_code == 200:
                st.success('YouTube video has been converted to MP3!')
            else:
                st.error("An error occurred during the conversion process")
                st.write(response.text)
    else:
        st.error("Please enter a valid YouTube URL.")
st.sidebar.divider()

# Selection box for available MP3 files in the 'downloads' folder
downloads_path = 'downloads'
files = os.listdir(downloads_path)
mp3_files = [file for file in files if file.endswith('.mp3')]
selected_file = st.sidebar.selectbox("Select an available MP3 file:", mp3_files)

# Button to trigger transcription of the selected MP3 file
if st.sidebar.button('Transcribe MP3'):
    if selected_file:
        with st.spinner('Transcribing...'):
            file_path = os.path.join(downloads_path, selected_file)
            files = {'file': (selected_file, open(file_path, 'rb'), 'audio/mpeg')}
            response = requests.post("http://localhost:8000/transcribe/", files=files)
            if response.status_code == 200:
                transcription = response.json()["transcription"]
                st.text_area("Transcription:", value=transcription, height=300)
            else:
                st.error("An error occurred during the transcription process")
                st.write(response.text)

if audio_file is not None:
    with st.spinner('Transcribing...'):
        files = {"file": (audio_file.name, audio_file, "audio/mpeg")}
        response = requests.post("http://localhost:8000/transcribe/", files=files)
        if response.status_code == 200:
            transcription = response.json()["transcription"]
            st.text_area("Transcription:", value=transcription, height=300)
        else:
            st.error("An error occurred during the transcription process")
            st.write(response.text)
st.sidebar.divider()

st.sidebar.info("Transcribe all MP3 files in the 'batch' folder. Place your custom .mp3 files there.")
if st.sidebar.button('Transcribe All MP3 Files in Batch'):
    with st.spinner('Transcribing batch...'):
        response = requests.get("http://localhost:8000/transcribe_batch/")

        if response.status_code == 200:
            transcriptions = response.json()
            for audio_file, transcription in transcriptions.items():
                st.subheader(f"Transcription for {audio_file}:")
                # Use the filename as a unique key for the text_area
                st.text_area("", value=transcription, height=150, key=audio_file)
                st.markdown("---")  # Separator
        else:
            st.error("An error occurred during the batch transcription process")
            st.write(response.text)


