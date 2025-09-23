"""
AI-Powered Job Interview Coach - NLP Analysis Page
Advanced NLP processing and visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ai_modules.nlp_processor import NLPProcessor
from ai_modules.auth import check_auth_status, init_session_state
from ai_modules.auth_ui import show_auth_page

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Interview Coach - NLP Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main NLP analysis function"""
    # Initialize session state
    init_session_state()
    
    # Check authentication
    is_authenticated = check_auth_status()
    if not is_authenticated:
        show_auth_page()
        return
    
    # Header
    st.markdown('<h1 class="main-header">📊 NLP Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("### 📝 Text Input")
    
    sample_texts = [
        "I am very excited about this opportunity and I believe I would be a great fit for this role. I have extensive experience in Python development and machine learning.",
        "Well, I think I'm pretty good at coding. I've done some projects and stuff. I know Python and JavaScript.",
        "In my previous role at TechCorp, I led a team of 5 developers to build a scalable microservices architecture that improved system performance by 40%.",
        "I'm not sure about this question. I guess I would try to figure it out as I go along."
    ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        text_choice = st.selectbox(
            "Choose sample text:",
            ["Custom Input"] + [f"Sample {i+1}" for i in range(len(sample_texts))],
            help="Select a pre-defined sample or use custom input"
        )
    
    with col2:
        if text_choice == "Custom Input":
            user_text = st.text_area(
                "Enter text to analyze:",
                value="",
                height=100,
                placeholder="Type or paste text here for NLP analysis...",
                help="Enter text for comprehensive NLP analysis"
            )
        else:
            sample_idx = int(text_choice.split()[-1]) - 1
            user_text = st.text_area(
                "Enter text to analyze:",
                value=sample_texts[sample_idx],
                height=100,
                help="Edit the sample text or use as is"
            )
    
    # Analyze button
    if st.button("🔍 Analyze Text", type="primary"):
        if not user_text.strip():
            st.error("Please enter text to analyze.")
        else:
            with st.spinner("🤖 Processing text with NLP..."):
                try:
                    # Initialize NLP processor
                    nlp_processor = NLPProcessor()
                    
                    # Process text
                    cleaned_data = nlp_processor.preprocess_text(user_text)
                    features = nlp_processor.extract_features(user_text, cleaned_data)
                    
                    # Display results
                    display_nlp_results(user_text, cleaned_data, features)
                    
                except Exception as e:
                    st.error(f"Error analyzing text: {str(e)}")


def display_nlp_results(original_text, cleaned_data, features):
    """Display comprehensive NLP analysis results"""
    
    # Overview metrics
    st.markdown("### 📈 Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Word Count",
            value=features['word_count'],
            help="Total number of words"
        )
    
    with col2:
        st.metric(
            label="Character Count",
            value=features['char_count'],
            help="Total number of characters"
        )
    
    with col3:
        st.metric(
            label="Reading Level",
            value=f"{features['flesch_reading_ease']:.1f}",
            help="Flesch Reading Ease Score"
        )
    
    with col4:
        st.metric(
            label="Sentiment",
            value=features['sentiment_label'],
            help="Overall sentiment analysis"
        )
    
    # Text processing results
    st.markdown("### 🔧 Text Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Text")
        st.text_area("", value=original_text, height=100, disabled=True)
    
    with col2:
        st.markdown("#### Cleaned Text (No Stopwords)")
        cleaned_text = ' '.join(cleaned_data['no_stopwords'])
        st.text_area("", value=cleaned_text, height=100, disabled=True)
    
    # Detailed features
    st.markdown("### 🔍 Detailed Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Keywords")
        keywords = ', '.join(features['keywords'][:10])  # Show top 10
        st.text_area("", value=keywords, height=80, disabled=True)
        
        st.markdown("#### Named Entities")
        entities = ', '.join([f"{ent.text} ({ent.label_})" for ent in features['entities'][:5]])
        st.text_area("", value=entities, height=80, disabled=True)
    
    with col2:
        st.markdown("#### Sentiment Scores")
        sentiment_data = {
            'Positive': features['sentiment_scores']['POSITIVE'],
            'Negative': features['sentiment_scores']['NEGATIVE']
        }
        
        fig = px.bar(
            x=list(sentiment_data.keys()),
            y=list(sentiment_data.values()),
            title="Sentiment Distribution",
            color=list(sentiment_data.keys()),
            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Text statistics
    st.markdown("### 📊 Text Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Readability")
        st.metric("Flesch Score", f"{features['flesch_reading_ease']:.1f}")
        st.metric("Grade Level", f"{features['flesch_grade_level']:.1f}")
    
    with col2:
        st.markdown("#### Complexity")
        st.metric("Avg Word Length", f"{features['avg_word_length']:.1f}")
        st.metric("Syllable Count", features['syllable_count'])
    
    with col3:
        st.markdown("#### Structure")
        st.metric("Sentence Count", features['sentence_count'])
        st.metric("Avg Words/Sentence", f"{features['avg_words_per_sentence']:.1f}")
    
    # Visualization
    st.markdown("### 📈 Visualizations")
    
    # Word frequency
    if features['keywords']:
        word_freq = features['keyword_frequencies']
        if word_freq:
            freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
            freq_df = freq_df.head(10)  # Top 10 words
            
            fig = px.bar(
                freq_df,
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top Keywords by Frequency",
                color='Frequency',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment over time (if multiple sentences)
    if features['sentence_count'] > 1:
        st.markdown("#### Sentiment Over Sentences")
        sentence_sentiments = []
        for i, sentence in enumerate(cleaned_data['sentences']):
            sent_features = nlp_processor.extract_features(sentence, nlp_processor.preprocess_text(sentence))
            sentence_sentiments.append({
                'Sentence': i+1,
                'Sentiment': sent_features['sentiment_scores']['POSITIVE'] - sent_features['sentiment_scores']['NEGATIVE']
            })
        
        if sentence_sentiments:
            sent_df = pd.DataFrame(sentence_sentiments)
            fig = px.line(
                sent_df,
                x='Sentence',
                y='Sentiment',
                title="Sentiment Trend Across Sentences",
                markers=True
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
