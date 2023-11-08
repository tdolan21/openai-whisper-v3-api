#!/bin/bash

# Execute the streamlit command
streamlit run app.py &

# Execute the uvicorn command
uvicorn api:app --reload
