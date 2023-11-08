# OpenAI Whisper-v3 API

## Introduction

First, make sure that your preffered torch environment is configured.

I generally use conda.

Then, you can install the requirements with:

```bash
git clone https://github.com/tdolan21/openai-whisper-v3-api
cd openai-whisper-v3-api
pip install -r requirements.txt
```

or 

```bash
pip install transformers datasets fastapi uvicorn pillow soundfile librosa pydub
```
## Usage

```bash
uvicorn api:app --reload
```
It will be available at localhost:8000


