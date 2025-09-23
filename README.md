# AI-Powered Job Interview Coach

An intelligent interview coaching application powered by advanced NLP and LLM technology.

## Features

- **🎭 Interview Simulation**: Dynamic question generation with AI-powered feedback
- **📊 NLP Analysis**: Comprehensive text analysis including sentiment, keywords, and readability
- **🤖 AI Evaluation**: Real-time scoring and suggestions for interview responses

## Pages

1. **Home**: Welcome page with feature overview
2. **Interview Simulation**: Practice with AI-generated questions
3. **NLP Analysis**: Deep text analysis and visualization

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT, Hugging Face Transformers, spaCy
- **NLP**: Text processing, sentiment analysis, keyword extraction
- **Authentication**: Supabase

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

3. Set environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Deployment

This app is optimized for deployment on Hugging Face Spaces and other cloud platforms.

## License

MIT License
