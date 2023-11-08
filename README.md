# OpenAI Whisper-v3 API

Welcome to the OpenAI Whisper-v3 API! This API leverages the power of OpenAI's Whisper model to transcribe audio into text. Before diving in, ensure that your preferred PyTorch environment is set up—Conda is recommended.

![Whisper-v3](assets/whisper-v3-thumbnail.png)

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.65.2-009688?style=flat&logo=fastapi)
![Librosa](https://img.shields.io/badge/Librosa-1.0.1-yellowgreen)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.11.8-red)

## Introduction


Clone and set up the repository as follows:

```bash
git clone https://github.com/tdolan21/openai-whisper-v3-api
cd openai-whisper-v3-api
pip install -r requirements.txt
chmod +x run.sh
```
Alternatively, install the dependencies directly:

```bash
pip install transformers datasets fastapi uvicorn pillow soundfile librosa pydub yt-dlp
```

You may need to upgrade to the dev version of transformers:

```bash
pip install transformers --upgrade
```

## Usage

```bash
./run.sh
```


## Features

+ Short-Form Transcription: Quick and efficient transcription for short audio clips.
+ Long-Form Transcription: Tailored for longer audio sessions, ensuring high accuracy.
+ Batch Transcription: Process multiple audio files simultaneously with ease.
+ YouTube to MP3: Extract and transcribe audio from YouTube videos directly.

### Batch Transcription

You can add any amount of mp3 files or subfolders containing .mp3 files and they will be normalized and transcribed with identifiers in the application.

For detailed usage and API endpoints, please refer to the API documentation once the server is running.


