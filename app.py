"""
AI-Powered Job Interview Coach - NLP Demo App
Simple Streamlit app to display NLP processing results from the notebook
"""

import streamlit as st
import pandas as pd
import json
from ai_modules.nlp_processor import process_interview_response, NLPProcessor
from ai_modules.auth import check_auth_status, init_session_state
from ai_modules.auth_ui import show_auth_page, show_logout_button, show_header_logout
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Interview Coach - NLP Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    init_session_state()
    
    # Check authentication status
    is_authenticated = check_auth_status()
    
    if not is_authenticated:
        show_auth_page()
        return
    
    # Header with logout button
    show_header_logout()
    
    # Show logout button in sidebar
    show_logout_button()
    
    # Sidebar for input
    st.sidebar.header("📝 Input Settings")
    
    # Sample questions
    sample_questions = [
        "Tell me about teamwork",
        "Describe a challenging project you worked on",
        "What are your technical skills?",
        "How do you handle conflicts in a team?",
        "Tell me about a time you failed and learned from it"
    ]
    
    selected_question = st.sidebar.selectbox(
        "Choose a sample question:",
        sample_questions
    )
    
    # Custom question input
    custom_question = st.sidebar.text_input(
        "Or enter your own question:",
        value=""
    )
    
    system_question = custom_question if custom_question else selected_question
    
    # Sample responses
    sample_responses = [
        "Umm I think I am good at teamwo rk, because in my last job I worked with a team of 5 people to build a Python application at Google.",
        "I'm really passionate about coding and I love working with JavaScript and React. I've built several web applications and I'm always learning new technologies.",
        "Well, I had this really difficult project where we had to optimize the database performance. It was challenging but I learned a lot about SQL and indexing.",
        "I think communication is key in any team. When there are disagreements, I try to listen to everyone's perspective and find a middle ground that works for everyone."
    ]
    
    st.sidebar.subheader("Sample Responses")
    sample_choice = st.sidebar.selectbox(
        "Choose a sample response:",
        ["Custom Input"] + [f"Sample {i+1}" for i in range(len(sample_responses))]
    )
    
    if sample_choice == "Custom Input":
        user_response = st.text_area(
            "Enter candidate response:",
            value="",
            height=100,
            placeholder="Type or paste the candidate's response here..."
        )
    else:
        sample_idx = int(sample_choice.split()[-1]) - 1
        user_response = st.text_area(
            "Enter candidate response:",
            value=sample_responses[sample_idx],
            height=100
        )
    
    # Process button
    if st.button("🔍 Analyze Response", type="primary"):
        if not user_response.strip():
            st.error("Please enter a response to analyze.")
        else:
            with st.spinner("Processing response..."):
                try:
                    # Process the response
                    result = process_interview_response(user_response, system_question)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        display_results(result, system_question)
                        
                except Exception as e:
                    st.error(f"Error processing response: {str(e)}")
                    st.exception(e)

def display_results(result, system_question):
    """Display the NLP analysis results in a structured way"""
    
    # Overview metrics
    st.markdown('<h2 class="section-header">📊 Analysis Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Score",
            value=f"{result['overall_score']}/3",
            delta=None
        )
    
    with col2:
        sentiment_color = "🟢" if result['sentiment_label'] == 'POSITIVE' else "🟡" if result['sentiment_label'] == 'NEUTRAL' else "🔴"
        st.metric(
            label="Sentiment",
            value=f"{sentiment_color} {result['sentiment_label']}",
            delta=f"{result['sentiment_score']:.2f}"
        )
    
    with col3:
        st.metric(
            label="Keywords Found",
            value=len(result['keywords']),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Named Entities",
            value=len(result['named_entities']),
            delta=None
        )
    
    # Rubric scores
    st.markdown('<h2 class="section-header">📋 Evaluation Rubric</h2>', unsafe_allow_html=True)
    
    rubric_cols = st.columns(3)
    rubric_colors = {
        'relevance': 'success' if result['rubric']['relevance'] else 'danger',
        'clarity': 'success' if result['rubric']['clarity'] else 'danger',
        'tone': 'success' if result['rubric']['tone'] else 'danger'
    }
    
    for i, (criterion, score) in enumerate(result['rubric'].items()):
        with rubric_cols[i]:
            status = "✅ Pass" if score else "❌ Fail"
            color_class = rubric_colors[criterion]
            st.markdown(f"""
            <div class="metric-card {color_class}">
                <strong>{criterion.title()}</strong><br>
                {status}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed analysis
    st.markdown('<h2 class="section-header">🔍 Detailed Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Text Processing", 
        "🏷️ Keywords & Entities", 
        "💭 Sentiment Analysis", 
        "📈 Visualization", 
        "📋 Raw Data"
    ])
    
    with tab1:
        st.subheader("Text Preprocessing Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Response:**")
            st.text(result['original_response'])
            
            st.write("**After Lowercase:**")
            st.text(result['preprocessing_steps']['lowercase'])
            
            st.write("**After Removing Fillers:**")
            st.text(result['preprocessing_steps']['no_fillers'])
        
        with col2:
            st.write("**After Removing Punctuation:**")
            st.text(result['preprocessing_steps']['no_punctuation'])
            
            st.write("**Tokenized Words:**")
            st.text(", ".join(result['tokenized_words']))
            
            st.write("**Lemmatized Words:**")
            st.text(", ".join(result['lemmatized_words']))
            
            st.write("**Final Cleaned (No Stopwords):**")
            st.text(", ".join(result['cleaned_response']))
    
    with tab2:
        st.subheader("Keywords and Named Entities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Extracted Keywords:**")
            if result['keywords']:
                for keyword in result['keywords']:
                    st.markdown(f"• {keyword}")
            else:
                st.write("No keywords found")
        
        with col2:
            st.write("**Named Entities:**")
            if result['named_entities']:
                for entity, label in result['named_entities']:
                    st.markdown(f"• **{entity}** ({label})")
            else:
                st.write("No named entities found")
    
    with tab3:
        st.subheader("Sentiment Analysis")
        
        sentiment_data = {
            'Sentiment': [result['sentiment_label']],
            'Confidence': [result['sentiment_score']]
        }
        
        df_sentiment = pd.DataFrame(sentiment_data)
        st.dataframe(df_sentiment, use_container_width=True)
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['sentiment_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Confidence"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data Visualization")
        
        # Rubric scores chart
        rubric_df = pd.DataFrame(list(result['rubric'].items()), columns=['Criterion', 'Score'])
        fig_rubric = px.bar(
            rubric_df, 
            x='Criterion', 
            y='Score',
            title="Rubric Scores",
            color='Score',
            color_continuous_scale=['red', 'green']
        )
        fig_rubric.update_layout(yaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig_rubric, use_container_width=True)
        
        # Keywords word cloud (simple text representation)
        if result['keywords']:
            st.write("**Keywords Frequency:**")
            keyword_counts = {}
            for keyword in result['keywords']:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            keyword_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])
            fig_keywords = px.bar(
                keyword_df, 
                x='Keyword', 
                y='Count',
                title="Keywords Frequency"
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
    
    with tab5:
        st.subheader("Raw Analysis Data")
        
        # Display raw JSON data
        st.json(result)
        
        # Download button for raw data
        json_str = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download Analysis Results",
            data=json_str,
            file_name="nlp_analysis_results.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
