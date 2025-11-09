#!/bin/bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Start Streamlit app
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
