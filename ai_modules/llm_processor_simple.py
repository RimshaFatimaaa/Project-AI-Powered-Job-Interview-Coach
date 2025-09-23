"""
Simplified LLM Processor Module - Step 2
Core features: Question generation, answer evaluation, and structured feedback
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

# LLM imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Types of interview questions"""
    HR_BEHAVIORAL = "hr_behavioral"
    TECHNICAL = "technical"

class DifficultyLevel(Enum):
    """Difficulty levels for questions"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class Question:
    """Structured question data"""
    question_text: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    expected_keywords: List[str] = None

@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    overall_score: float  # 0-100
    relevance_score: float  # 0-100
    clarity_score: float  # 0-100
    correctness_score: float  # 0-100
    feedback: str
    suggestions: List[str]

class SimpleLLMProcessor:
    """Simplified LLM processor for core interview features"""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        """
        Initialize simplified LLM processor
        
        Args:
            use_openai: Whether to use OpenAI API (True) or local models (False)
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.use_openai = use_openai
        self.openai_client = None
        self.local_llm = None
        
        # Initialize based on preference
        if use_openai:
            self._init_openai(openai_api_key)
        else:
            self._init_local_models()
        
        # Canonical answers database (manual for now)
        self.canonical_answers = {
            "teamwork": "Teamwork involves collaborating effectively with others to achieve common goals. It requires communication, active listening, conflict resolution, and supporting team members. A good team player contributes ideas, helps others when needed, and maintains a positive attitude.",
            "leadership": "Leadership is the ability to guide and inspire others toward achieving shared objectives. It involves setting clear goals, making decisions, motivating team members, providing feedback, and leading by example. Effective leaders communicate vision, delegate tasks appropriately, and support team development.",
            "problem_solving": "Problem-solving is the process of identifying, analyzing, and resolving issues systematically. It involves defining the problem clearly, gathering relevant information, generating multiple solutions, evaluating options, implementing the best solution, and monitoring results for continuous improvement.",
            "technical_skills": "Technical skills refer to the specific knowledge and abilities required to perform job-related tasks. This includes programming languages, software tools, methodologies, and domain expertise. Continuous learning and staying updated with industry trends are essential for maintaining technical competency.",
            "communication": "Communication is the ability to convey information clearly and effectively. It includes verbal, written, and non-verbal communication skills. Good communicators listen actively, ask clarifying questions, adapt their message to the audience, and provide constructive feedback.",
            "adaptability": "Adaptability is the ability to adjust to new conditions, environments, or challenges. It involves being flexible, open to change, learning new skills quickly, and maintaining performance under pressure. Adaptable individuals embrace uncertainty and view change as an opportunity for growth."
        }
        
        # Interview session state
        self.current_session = {
            "questions_asked": [],
            "current_difficulty": DifficultyLevel.MEDIUM,
            "session_score": 0.0,
            "total_questions": 0
        }

    def _init_openai(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        try:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found. Falling back to local models.")
                self.use_openai = False
                self._init_local_models()
                return
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.use_openai = False
            self._init_local_models()

    def _init_local_models(self):
        """Initialize local Hugging Face models"""
        try:
            # Skip local model initialization on Render to avoid memory issues
            logger.warning("Skipping local model initialization - using template-based processing")
            self.local_llm = None
            self.tokenizer = None
            self.model = None
            logger.info("Local models skipped - using fallback processing")
        except Exception as e:
            logger.error(f"Failed to initialize local models: {e}")
            self.local_llm = None

    def generate_question(
        self, 
        question_type: QuestionType, 
        role: str = "Software Engineer",
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    ) -> Question:
        """
        Generate a dynamic interview question
        
        Args:
            question_type: Type of question to generate
            role: Job role/position
            difficulty: Difficulty level
            
        Returns:
            Generated Question object
        """
        try:
            # Simple question templates
            if question_type == QuestionType.HR_BEHAVIORAL:
                topics = ["teamwork", "leadership", "problem solving", "communication", "adaptability"]
                topic = topics[len(self.current_session["questions_asked"]) % len(topics)]
                question_text = f"Tell me about a time when you demonstrated {topic} in your previous role."
                expected_keywords = [topic, "experience", "situation", "result", "learned"]
            else:  # TECHNICAL
                topics = ["programming", "problem solving", "technical challenges", "learning", "projects"]
                topic = topics[len(self.current_session["questions_asked"]) % len(topics)]
                question_text = f"Describe a {topic} project you worked on and the technical challenges you faced."
                expected_keywords = [topic, "project", "technical", "challenges", "solution", "technologies"]
            
            # Adjust difficulty
            if difficulty == DifficultyLevel.EASY:
                question_text += " Please provide a brief overview."
            elif difficulty == DifficultyLevel.HARD:
                question_text += " Please provide specific details about your approach, technologies used, and lessons learned."
            
            # Create question object
            question = Question(
                question_text=question_text,
                question_type=question_type,
                difficulty=difficulty,
                expected_keywords=expected_keywords
            )
            
            # Update session
            self.current_session["questions_asked"].append(question_text)
            self.current_session["total_questions"] += 1
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            # Return fallback question
            return Question(
                question_text="Tell me about yourself and your relevant experience.",
                question_type=question_type,
                difficulty=difficulty,
                expected_keywords=["experience", "skills", "background"]
            )

    def evaluate_answer(
        self, 
        question: str, 
        candidate_answer: str,
        cleaned_answer: str = None
    ) -> EvaluationResult:
        """
        Evaluate candidate's answer against rubric
        
        Args:
            question: The interview question
            candidate_answer: Original candidate's answer
            cleaned_answer: Cleaned answer from NLP module (optional)
            
        Returns:
            EvaluationResult with scores and feedback
        """
        try:
            # Use cleaned answer if provided, otherwise use original
            answer_to_evaluate = cleaned_answer if cleaned_answer else candidate_answer
            
            # Simple keyword-based evaluation
            question_lower = question.lower()
            answer_lower = answer_to_evaluate.lower()
            
            # Extract topic from question
            topic = None
            for key in self.canonical_answers.keys():
                if key in question_lower:
                    topic = key
                    break
            
            # Get canonical answer for comparison
            canonical_answer = self.canonical_answers.get(topic, "")
            
            # Calculate scores
            relevance_score = self._calculate_relevance_score(question_lower, answer_lower)
            clarity_score = self._calculate_clarity_score(answer_to_evaluate)
            correctness_score = self._calculate_correctness_score(answer_lower, canonical_answer)
            
            # Calculate overall score
            overall_score = (relevance_score + clarity_score + correctness_score) / 3
            
            # Generate feedback
            feedback = self._generate_feedback(relevance_score, clarity_score, correctness_score, topic)
            suggestions = self._generate_suggestions(relevance_score, clarity_score, correctness_score)
            
            # Update session score
            if self.current_session["total_questions"] > 0:
                self.current_session["session_score"] = (
                    (self.current_session["session_score"] * (self.current_session["total_questions"] - 1) + overall_score) 
                    / self.current_session["total_questions"]
                )
            else:
                self.current_session["session_score"] = overall_score
            
            return EvaluationResult(
                overall_score=overall_score,
                relevance_score=relevance_score,
                clarity_score=clarity_score,
                correctness_score=correctness_score,
                feedback=feedback,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return self._create_fallback_evaluation()

    def _calculate_relevance_score(self, question: str, answer: str) -> float:
        """Calculate relevance score based on keyword matching"""
        # Extract key terms from question
        question_terms = set(question.split())
        
        # Check if answer addresses the question
        answer_terms = set(answer.split())
        common_terms = question_terms.intersection(answer_terms)
        
        # Basic relevance scoring
        if len(common_terms) > 3:
            return 85.0
        elif len(common_terms) > 1:
            return 70.0
        else:
            return 50.0

    def _calculate_clarity_score(self, answer: str) -> float:
        """Calculate clarity score based on answer structure"""
        # Simple clarity metrics
        word_count = len(answer.split())
        
        if word_count > 50:
            return 80.0
        elif word_count > 20:
            return 70.0
        else:
            return 60.0

    def _calculate_correctness_score(self, answer: str, canonical: str) -> float:
        """Calculate correctness score based on canonical answer comparison"""
        if not canonical:
            return 70.0  # Default score if no canonical answer
        
        # Simple keyword matching with canonical answer
        canonical_terms = set(canonical.lower().split())
        answer_terms = set(answer.split())
        common_terms = canonical_terms.intersection(answer_terms)
        
        if len(common_terms) > 5:
            return 85.0
        elif len(common_terms) > 2:
            return 70.0
        else:
            return 55.0

    def _generate_feedback(self, relevance: float, clarity: float, correctness: float, topic: str) -> str:
        """Generate feedback based on scores"""
        feedback_parts = []
        
        if relevance >= 80:
            feedback_parts.append("Your answer directly addresses the question asked.")
        elif relevance >= 60:
            feedback_parts.append("Your answer partially addresses the question.")
        else:
            feedback_parts.append("Your answer could be more relevant to the specific question.")
        
        if clarity >= 80:
            feedback_parts.append("Your response is clear and well-structured.")
        elif clarity >= 60:
            feedback_parts.append("Your response is generally clear but could be more detailed.")
        else:
            feedback_parts.append("Your response could be clearer and more detailed.")
        
        if correctness >= 80:
            feedback_parts.append("Your answer demonstrates good understanding of the topic.")
        elif correctness >= 60:
            feedback_parts.append("Your answer shows some understanding but could be more accurate.")
        else:
            feedback_parts.append("Consider providing more accurate information about the topic.")
        
        return " ".join(feedback_parts)

    def _generate_suggestions(self, relevance: float, clarity: float, correctness: float) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if relevance < 70:
            suggestions.append("Make sure to directly address the specific question asked.")
        
        if clarity < 70:
            suggestions.append("Provide more specific examples and details in your response.")
        
        if correctness < 70:
            suggestions.append("Consider researching the topic more thoroughly for future responses.")
        
        if not suggestions:
            suggestions.append("Continue providing detailed, relevant examples in your responses.")
        
        return suggestions

    def _create_fallback_evaluation(self) -> EvaluationResult:
        """Create fallback evaluation when processing fails"""
        return EvaluationResult(
            overall_score=50.0,
            relevance_score=50.0,
            clarity_score=50.0,
            correctness_score=50.0,
            feedback="Unable to provide detailed evaluation. Please try again.",
            suggestions=["Ensure your answer is clear and relevant to the question."]
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get current interview session summary"""
        return {
            "total_questions": self.current_session["total_questions"],
            "average_score": self.current_session["session_score"],
            "current_difficulty": self.current_session["current_difficulty"].value,
            "questions_asked": self.current_session["questions_asked"]
        }

    def adjust_difficulty(self, performance_score: float):
        """Adjust difficulty based on performance"""
        if performance_score >= 80:
            self.current_session["current_difficulty"] = DifficultyLevel.HARD
        elif performance_score >= 60:
            self.current_session["current_difficulty"] = DifficultyLevel.MEDIUM
        else:
            self.current_session["current_difficulty"] = DifficultyLevel.EASY

# Convenience functions
def generate_question(
    question_type: str = "hr_behavioral",
    role: str = "Software Engineer",
    difficulty: str = "medium"
) -> Question:
    """Convenience function to generate a single question"""
    processor = SimpleLLMProcessor(use_openai=False)
    return processor.generate_question(
        QuestionType(question_type),
        role,
        DifficultyLevel(difficulty)
    )

def evaluate_answer(question: str, answer: str, cleaned_answer: str = None) -> EvaluationResult:
    """Convenience function to evaluate an answer"""
    processor = SimpleLLMProcessor(use_openai=False)
    return processor.evaluate_answer(question, answer, cleaned_answer)
