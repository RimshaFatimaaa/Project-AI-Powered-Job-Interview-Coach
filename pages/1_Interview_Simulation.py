"""
AI-Powered Job Interview Coach - Interview Simulation Page
Main interview simulation functionality
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from ai_modules.nlp_processor import process_interview_response, NLPProcessor
from ai_modules.llm_processor_simple import SimpleLLMProcessor, QuestionType, DifficultyLevel
from ai_modules.auth import check_auth_status, init_session_state
from ai_modules.auth_ui import show_auth_page, show_logout_button, show_header_logout
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Interview Coach - Interview Simulation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.5rem;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo-icon {
        font-size: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .logo-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .logout-btn {
        background: #ffffff;
        color: #6c757d;
        border: 1px solid #dee2e6;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .logout-btn:hover {
        background: #f8f9fa;
        border-color: #adb5bd;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Input section styling */
    .input-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    .mode-selector {
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: #6c757d;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: #5a6268;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: #f8f9fa;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    .footer h4 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .footer p {
        color: #6c757d;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main interview simulation function"""
    # Initialize session state
    init_session_state()
    
    # Check authentication
    is_authenticated = check_auth_status()
    if not is_authenticated:
        show_auth_page()
        return
    
    # Custom header with logo
    st.markdown("""
    <div class="header-container">
        <div class="logo-section">
            <div class="logo-icon">🎯</div>
            <div>
                <h1 class="logo-text">AI Interview Coach</h1>
                <p style="color: #6c757d; margin: 0; font-size: 0.85rem; font-weight: 400;">Smart Analysis • AI-Powered</p>
            </div>
        </div>
        <div>
            <button class="logout-btn" onclick="window.location.href='?logout=true'">Logout</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<h1 class="main-header">🎭 Interview Simulation</h1>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Interview Simulation Mode
    st.markdown("### 🎭 Interview Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        question_type = st.selectbox(
            "Question Type:",
            ["hr_behavioral", "technical"],
            help="Select the type of question"
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty Level:",
            ["easy", "medium", "hard"],
            help="Choose the difficulty level"
        )
    
    # Generate question for simulation
    if st.button("🎯 Generate Question", type="primary"):
        with st.spinner("Generating question..."):
            try:
                llm_processor = SimpleLLMProcessor(use_openai=True)
                question = llm_processor.generate_question(
                    QuestionType(question_type),
                    "Software Engineer",
                    DifficultyLevel(difficulty)
                )
                st.session_state['current_question'] = question
                st.session_state['llm_processor'] = llm_processor
            except Exception as e:
                st.error(f"Error generating question: {str(e)}")
    
    # Display current question
    if 'current_question' in st.session_state:
        question = st.session_state['current_question']
        st.markdown(f"**Question:** {question.question_text}")
        st.markdown(f"**Type:** {question.question_type.value}")
        st.markdown(f"**Difficulty:** {question.difficulty.value}")
        system_question = question.question_text
    
    # Response input section
    st.markdown("### 💬 Response Input")
    
    sample_responses = [
        "Umm I think I am good at teamwo rk, because in my last job I worked with a team of 5 people to build a Python application at Google.",
        "I'm really passionate about coding and I love working with JavaScript and React. I've built several web applications and I'm always learning new technologies.",
        "Well, I had this really difficult project where we had to optimize the database performance. It was challenging but I learned a lot about SQL and indexing.",
        "I think communication is key in any team. When there are disagreements, I try to listen to everyone's perspective and find a middle ground that works for everyone."
    ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        sample_choice = st.selectbox(
        "Choose a sample response:",
            ["Custom Input"] + [f"Sample {i+1}" for i in range(len(sample_responses))],
            help="Select a pre-defined sample response or use custom input"
    )
    
    with col2:
        if sample_choice == "Custom Input":
            user_response = st.text_area(
                "Enter candidate response:",
                value="",
                height=100,
                placeholder="Type or paste the candidate's response here...",
                help="Enter the candidate's response to analyze"
            )
        else:
            sample_idx = int(sample_choice.split()[-1]) - 1
            user_response = st.text_area(
                "Enter candidate response:",
                value=sample_responses[sample_idx],
                height=100,
                help="Edit the sample response or use as is"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close input section
    
    # Process button
    if st.button("🔍 Analyze Response", type="primary"):
        if not user_response.strip():
            st.error("Please enter a response to analyze.")
        else:
            with st.spinner("🤖 Analyzing response..."):
                try:
                    # First run NLP preprocessing
                    nlp_processor = NLPProcessor()
                    cleaned_data = nlp_processor.preprocess_text(user_response)
                    features = nlp_processor.extract_features(user_response, cleaned_data)
                    
                    # Then run LLM evaluation with cleaned text
                    if 'llm_processor' in st.session_state:
                        llm_processor = st.session_state['llm_processor']
                    else:
                        llm_processor = SimpleLLMProcessor(use_openai=True)
                    
                    evaluation = llm_processor.evaluate_answer(
                        question=system_question,
                        candidate_answer=user_response,
                        cleaned_answer=' '.join(cleaned_data['no_stopwords'])
                    )
                    
                    # Display results
                    display_llm_evaluation(evaluation, system_question)
                    
                    # Show session summary
                    session_summary = llm_processor.get_session_summary()
                    display_session_summary(session_summary)
                    
                    # Adjust difficulty for next question
                    llm_processor.adjust_difficulty(evaluation.overall_score)
                        
                except Exception as e:
                    st.error(f"Error analyzing response: {str(e)}")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h4>🤖 AI-Powered Job Interview Coach</h4>
        <p>Powered by Advanced NLP & LLM Technology | Step 2: Dynamic Question Generation & Answer Evaluation</p>
        <p>Built with ❤️ using Streamlit, Transformers, and LangChain</p>
    </div>
    """, unsafe_allow_html=True)


def display_session_summary(session_summary):
    """Display interview session summary"""
    st.markdown("### 📊 Session Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Questions Asked",
            value=session_summary["total_questions"]
        )
    
    with col2:
        st.metric(
            label="Average Score",
            value=f"{session_summary['average_score']:.1f}/100"
        )
    
    with col3:
        st.metric(
            label="Current Difficulty",
            value=session_summary["current_difficulty"].title()
        )


def display_llm_evaluation(evaluation, question):
    """Display simplified LLM evaluation results"""
    st.markdown('<h2 class="section-header">🤖 LLM Evaluation Results</h2>', unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Score",
            value=f"{evaluation.overall_score:.1f}/100",
            delta=f"{evaluation.overall_score - 50:.1f}",
            help="Overall performance score"
        )
    
    with col2:
        st.metric(
            label="Relevance",
            value=f"{evaluation.relevance_score:.1f}/100",
            help="How well the answer addresses the question"
        )
    
    with col3:
        st.metric(
            label="Clarity",
            value=f"{evaluation.clarity_score:.1f}/100",
            help="How clear and well-structured the answer is"
        )
    
    with col4:
        st.metric(
            label="Correctness",
            value=f"{evaluation.correctness_score:.1f}/100",
            help="Technical accuracy and correctness"
        )
    
    # Feedback
    st.markdown("#### 💬 Feedback")
    st.info(evaluation.feedback)
    
    # Suggestions
    st.markdown("#### 💡 Suggestions")
    for i, suggestion in enumerate(evaluation.suggestions, 1):
        st.markdown(f"{i}. {suggestion}")


if __name__ == "__main__":
    main()
