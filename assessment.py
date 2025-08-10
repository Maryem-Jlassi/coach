import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"  # FIX: add ollama/ prefix
    API_BASE = "http://localhost:11434"
    CRISIS_KEYWORDS = ["suicide", "kill myself", "end my life", "hurt myself", "self harm", "die"]

class AgeGroup(Enum):
    CHILD = "8-12"
    TEENAGER = "13-17"
    YOUNG_ADULT = "18-25"
    ADULT = "26-64"
    SENIOR = "65+"

class RiskLevel(Enum):
    LOW = "green"
    MODERATE = "yellow"
    HIGH = "orange"
    CRISIS = "red"

@dataclass
class UserSession:
    session_id: str
    age: int
    age_group: AgeGroup
    responses: Dict[str, Any]
    risk_level: Optional[RiskLevel] = None
    analysis: Optional[str] = None
    recommendations: List[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recommendations is None:
            self.recommendations = []

class TemporarySessionManager:
    def __init__(self):
        self._active_sessions = {}
        self._session_timeout = 3600

    def create_temporary_session(self, session: UserSession) -> str:
        self._active_sessions[session.session_id] = {
            'session': session,
            'created_at': datetime.now(),
            'last_accessed': datetime.now()
        }
        return session.session_id

    def get_session(self, session_id: str) -> Optional[UserSession]:
        if session_id in self._active_sessions:
            session_data = self._active_sessions[session_id]
            session_data['last_accessed'] = datetime.now()
            return session_data['session']
        return None

    def update_session(self, session: UserSession):
        if session.session_id in self._active_sessions:
            self._active_sessions[session.session_id]['session'] = session
            self._active_sessions[session.session_id]['last_accessed'] = datetime.now()

    def clear_session(self, session_id: str):
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]['session']
            session.responses.clear()
            session.analysis = None
            del self._active_sessions[session_id]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        if session:
            return {
                'session_id': session.session_id,
                'age_group': session.age_group.value,
                'risk_level': session.risk_level.value if session.risk_level else None,
                'timestamp': session.timestamp.isoformat(),
                'has_analysis': bool(session.analysis),
                'has_recommendations': bool(session.recommendations)
            }
        return {}

class CrisisDetector:
    @staticmethod
    def detect_crisis(responses: Dict[str, str]) -> bool:
        for response in responses.values():
            if isinstance(response, str):
                response_lower = response.lower()
                for keyword in Config.CRISIS_KEYWORDS:
                    if keyword in response_lower:
                        return True
        return False

    @staticmethod
    def get_crisis_resources() -> Dict[str, str]:
        return {
            "National Suicide Prevention Lifeline": "988",
            "Crisis Text Line": "Text HOME to 741741",
            "Emergency Services": "911",
            "International Association for Suicide Prevention": "https://www.iasp.info/resources/Crisis_Centres/"
        }

class MentalHealthDiseaseClassifier:
    def __init__(self, llm):
        self.llm = llm

    def classify_mental_health_condition(self, responses: Dict[str, str], age_group: AgeGroup) -> Dict[str, Any]:
        if CrisisDetector.detect_crisis(responses):
            return {
                'risk_level': RiskLevel.CRISIS,
                'primary_concerns': ['Crisis - Immediate intervention needed'],
                'condition_type': 'Crisis',
                'severity': 'Critical',
                'confidence': 'High',
                'key_symptoms': [],
                'protective_factors': []
            }
        responses_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in responses.items()])
        prompt = f"""
        You are a mental health professional analyzing responses from a {age_group.value} year old person.

        Analyze these responses and identify potential mental health patterns:

        {responses_text}

        Based on these responses, provide a structured analysis in this format:
        RISK_LEVEL: [level]
        PRIMARY_CONCERNS: [list concerns]
        CONDITION_TYPE: [type]
        SEVERITY: [severity]
        CONFIDENCE: [confidence]
        KEY_SYMPTOMS: [symptoms]
        PROTECTIVE_FACTORS: [factors]
        """
        analysis_result = self.llm.call(prompt)  # FIX: use .call for CrewAI LLM
        return self._parse_classification_result(analysis_result)

    def _parse_classification_result(self, result: str) -> Dict[str, Any]:
        classification = {
            'risk_level': RiskLevel.LOW,
            'primary_concerns': [],
            'condition_type': 'Assessment needed',
            'severity': 'Unknown',
            'confidence': 'Medium',
            'key_symptoms': [],
            'protective_factors': []
        }
        lines = result.strip().split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if 'risk_level' in key:
                    risk_mapping = {
                        'low': RiskLevel.LOW,
                        'moderate': RiskLevel.MODERATE,
                        'high': RiskLevel.HIGH,
                        'crisis': RiskLevel.CRISIS
                    }
                    classification['risk_level'] = risk_mapping.get(value.lower(), RiskLevel.LOW)
                elif 'primary_concerns' in key:
                    classification['primary_concerns'] = [concern.strip() for concern in value.split(',')]
                elif 'condition_type' in key:
                    classification['condition_type'] = value
                elif 'severity' in key:
                    classification['severity'] = value
                elif 'confidence' in key:
                    classification['confidence'] = value
                elif 'key_symptoms' in key:
                    classification['key_symptoms'] = [symptom.strip() for symptom in value.split(',')]
                elif 'protective_factors' in key:
                    classification['protective_factors'] = [factor.strip() for factor in value.split(',')]
        return classification

class VirtualTherapistAgents:
    def __init__(self, model_name: str = Config.MODEL_NAME, api_base: str = Config.API_BASE):
        self.llm = LLM(model=model_name, base_url=api_base)  # FIX: use CrewAI LLM

    def create_intake_agent(self) -> Agent:
        return Agent(
            role='Virtual Therapist -  SafeMate ',
            goal='Create a safe, welcoming environment and gather initial user information',
            backstory='You are  SafeMate , a compassionate AI mental health intake specialist. Always introduce yourself as MediMate when talking to users.',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_question_generator_agent(self) -> Agent:
        return Agent(
            role='Question Generator',
            goal='Generate appropriate mental health screening questions for different age groups',
            backstory='You are an expert in psychological assessment.',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_analysis_agent(self) -> Agent:
        return Agent(
            role='Mental Health Analyst',
            goal='Analyze responses to identify mental health patterns and risk levels',
            backstory='You are an experienced clinical psychologist.',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def create_counselor_agent(self) -> Agent:
        return Agent(
            role='Virtual Counselor',
            goal='Provide empathetic analysis and practical recommendations',
            backstory='You are a warm, empathetic counselor.',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

class TaskTemplates:
    @staticmethod
    def create_intake_task(user_age: int, agent: Agent) -> Task:
        age_group = TaskTemplates.determine_age_group(user_age)
        prompt = f'''
        You are conducting an intake session for a {user_age}-year-old person (age group: {age_group.value}).
        Always introduce yourself as  SafeMate , a friendly AI assistant.
        Your tasks:
        1. Welcome them warmly and explain that you're here to listen and help
        2. Explain that this is a safe space and their responses are confidential
        3. Let them know that you're an AI assistant, not a human therapist
        4. Explain the process: you'll ask some questions to understand how they're feeling
        5. Get their consent to proceed
        6. Ask if they have any questions before starting
        Adapt your language and tone to be age-appropriate for a {age_group.value} year old.
        Be warm, professional, and reassuring.
        Return a welcome message and confirmation that they're ready to proceed.
        '''
        return Task(
            description=prompt,
            expected_output="A warm welcome message with process explanation and consent confirmation",
            agent=agent
        )

    @staticmethod
    def create_question_generation_task(age_group: AgeGroup, num_questions: int, agent: Agent) -> Task:
        prompt = f'''
        Generate {num_questions} unique, professional mental health screening questions for {age_group.value} year olds.
        Each question must end with a question mark.
        '''
        return Task(
            description=prompt,
            expected_output=f"A numbered list of exactly {num_questions} age-appropriate mental health screening questions",
            agent=agent
        )

    @staticmethod
    def create_analysis_task(session: UserSession, agent: Agent) -> Task:
        prompt = f'''
        Analyze the mental health screening responses for a {session.age}-year-old person.
        Responses received:
        {json.dumps(session.responses, indent=2)}
        Your analysis should identify:
        - Key Observations
        - Strengths Identified
        - Areas of Concern
        - Risk Assessment
        - Recommendations
        '''
        return Task(
            description=prompt,
            expected_output="Comprehensive mental health analysis with risk assessment and recommendations",
            agent=agent
        )

    @staticmethod
    def create_bilan_task(session: UserSession, analysis_result: str, agent: Agent) -> Task:
        prompt = f'''
        Create a compassionate summary ("bilan") for a {session.age}-year-old person based on their mental health screening.
        Analysis provided:
        {analysis_result}
        Risk level identified: {session.risk_level.value if session.risk_level else 'Not determined'}
        Include:
        1. PERSONAL STRENGTHS
        2. MENTAL HEALTH INSIGHTS
        3. CURRENT CHALLENGES
        4. RECOMMENDATIONS
        5. NEXT STEPS
        '''
        return Task(
            description=prompt,
            expected_output="A compassionate, comprehensive bilan summary.",
            agent=agent
        )

    @staticmethod
    def determine_age_group(age: int) -> AgeGroup:
        if 8 <= age <= 12:
            return AgeGroup.CHILD
        elif 13 <= age <= 17:
            return AgeGroup.TEENAGER
        elif 18 <= age <= 25:
            return AgeGroup.YOUNG_ADULT
        elif 26 <= age <= 64:
            return AgeGroup.ADULT
        else:
            return AgeGroup.SENIOR

class VirtualTherapistSystem:
    def __init__(self):
        self.session_manager = TemporarySessionManager()
        self.agents = VirtualTherapistAgents()
        self.classifier = MentalHealthDiseaseClassifier(self.agents.llm)

    def start_session(self, age: int) -> UserSession:
        session_id = str(uuid.uuid4())
        age_group = TaskTemplates.determine_age_group(age)
        session = UserSession(
            session_id=session_id,
            age=age,
            age_group=age_group,
            responses={}
        )
        self.session_manager.create_temporary_session(session)
        return session

    def conduct_intake(self, session: UserSession) -> str:
        intake_agent = self.agents.create_intake_agent()
        intake_task = TaskTemplates.create_intake_task(session.age, agent=intake_agent)
        crew = Crew(
            agents=[intake_agent],
            tasks=[intake_task],
            process=Process.sequential
        )
        result = crew.kickoff()
        return result

    def conduct_screening(self, session: UserSession) -> List[str]:
        question_agent = self.agents.create_question_generator_agent()
        question_task = TaskTemplates.create_question_generation_task(session.age_group, 15, agent=question_agent)
        crew = Crew(
            agents=[question_agent],
            tasks=[question_task],
            process=Process.sequential
        )
        questions_result = crew.kickoff()
        questions = self.parse_questions_from_result(questions_result)
        return questions

    def parse_questions_from_result(self, questions_result: str) -> List[str]:
        questions = []
        lines = str(questions_result).strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                if '.' in line:
                    question = line.split('.', 1)[-1].strip()
                else:
                    question = line.strip('•-').strip()
                if question and question.endswith('?'):
                    questions.append(question)
        if not questions:
            for line in lines:
                line = line.strip()
                if line.endswith('?'):
                    questions.append(line)
        return questions[:15]

    def process_user_responses(self, session: UserSession, responses: Dict[str, str]) -> UserSession:
        session.responses = responses
        self.session_manager.update_session(session)
        return session

    def analyze_and_classify(self, session: UserSession) -> UserSession:
        classification = self.classifier.classify_mental_health_condition(session.responses, session.age_group)
        session.risk_level = classification['risk_level']
        analysis_agent = self.agents.create_analysis_agent()
        analysis_task = TaskTemplates.create_analysis_task(session, agent=analysis_agent)
        crew = Crew(
            agents=[analysis_agent],
            tasks=[analysis_task],
            process=Process.sequential
        )
        analysis_result = crew.kickoff()
        session.analysis = str(analysis_result)
        session.recommendations = self.generate_recommendations(
            session.risk_level, session.age_group,
            classification['condition_type'],
            classification['severity']
        )
        self.session_manager.update_session(session)
        session.responses = {}
        return session

    def generate_bilan(self, session: UserSession) -> UserSession:
        counselor_agent = self.agents.create_counselor_agent()
        bilan_task = TaskTemplates.create_bilan_task(session, session.analysis, agent=counselor_agent)
        crew = Crew(
            agents=[counselor_agent],
            tasks=[bilan_task],
            process=Process.sequential
        )
        bilan_result = crew.kickoff()
        session.analysis = str(bilan_result)
        self.session_manager.update_session(session)
        return session

    def complete_session(self, session_id: str):
        self.session_manager.clear_session(session_id)

    def generate_recommendations(self, risk_level: RiskLevel, age_group: AgeGroup,
                                condition_type: str = None, severity: str = None) -> List[str]:
        base_recommendations = {
            RiskLevel.LOW: [
                "Continue practicing healthy coping strategies",
                "Maintain regular sleep and exercise routines",
                "Stay connected with supportive friends and family",
                "Consider trying mindfulness or relaxation techniques"
            ],
            RiskLevel.MODERATE: [
                "Consider speaking with a counselor or therapist",
                "Practice stress management techniques daily",
                "Prioritize self-care activities",
                "Reach out to trusted friends or family when struggling"
            ],
            RiskLevel.HIGH: [
                "Schedule an appointment with a mental health professional within the next week",
                "Create a support network of trusted people you can contact",
                "Develop a crisis plan for difficult moments",
                "Consider joining a support group"
            ],
            RiskLevel.CRISIS: [
                "IMMEDIATE: Contact crisis hotline or emergency services",
                "Go to nearest emergency room if having thoughts of self-harm",
                "Call National Suicide Prevention Lifeline: 988",
                "Contact a trusted friend or family member to stay with you"
            ]
        }
        age_specific = {
            AgeGroup.CHILD: ["Talk to a trusted adult like a parent, teacher, or school counselor"],
            AgeGroup.TEENAGER: ["Consider talking to a school counselor or trusted adult"],
            AgeGroup.YOUNG_ADULT: ["Look into counseling services at your school or workplace"],
            AgeGroup.ADULT: ["Consider employee assistance programs or community mental health services"],
            AgeGroup.SENIOR: ["Contact your primary care physician or local senior services"]
        }
        recommendations = base_recommendations[risk_level].copy()
        recommendations.extend(age_specific.get(age_group, []))
        if condition_type and condition_type.lower() != 'no significant concerns':
            recommendations.append(f"Condition-specific advice for {condition_type}, severity: {severity}")
        return recommendations

    def get_crisis_resources_display(self) -> str:
        resources = CrisisDetector.get_crisis_resources()
        formatted = "🚨 **CRISIS RESOURCES - GET HELP NOW** 🚨\n\n"
        for resource, contact in resources.items():
            formatted += f"• **{resource}**: {contact}\n"
        return formatted

def main():
    st.set_page_config(page_title="Virtual Mental Health Therapist", page_icon="🧠", layout="wide")
    st.title("🧠 Virtual Mental Health Therapist")
    st.markdown("*A compassionate AI assistant for mental health screening and support*")
    if 'session' not in st.session_state:
        st.session_state.session = None
    if 'stage' not in st.session_state:
        st.session_state.stage = 'start'
    @st.cache_resource
    def init_system():
        return VirtualTherapistSystem()
    system = init_system()
    with st.sidebar:
        st.header("Important Information")
        st.warning("""
        🚨 **Crisis Resources**
        - National Suicide Prevention Lifeline: 988
        - Crisis Text Line: Text HOME to 741741
        - Emergency: 911
        """)
        st.info("""
        📋 **What to Expect**
        1. Age verification
        2. Brief questionnaire (15 questions)
        3. Mental health analysis
        4. Personalized recommendations
        5. Resource suggestions
        """)
        st.markdown("""
        ⚠️ **Important Disclaimer**
        This AI assistant provides support and information but is NOT a substitute
        for professional mental health care. Always consult qualified professionals
        for serious mental health concerns.
        """)
    if st.session_state.stage == 'start':
        st.header("Welcome - Let's Begin")
        st.markdown("""
        I'm here to listen and provide support. This assessment will help us understand
        how you're feeling and provide personalized recommendations.
        Everything you share is confidential and will be used only to help you.
        """)
        age = st.number_input("What is your age?", min_value=8, max_value=100, value=25)
        if st.button("Start Assessment", type="primary"):
            st.session_state.session = system.start_session(age)
            st.session_state.stage = 'intake'
            st.rerun()
    elif st.session_state.stage == 'intake':
        st.header("Getting Started")
        with st.spinner("Preparing your personalized assessment..."):
            intake_result = system.conduct_intake(st.session_state.session)
        st.success(intake_result)
        if st.button("I'm Ready to Continue", type="primary"):
            st.session_state.stage = 'questions'
            st.rerun()
    elif st.session_state.stage == 'questions':
        st.header("Mental Health Assessment")
        session = st.session_state.session
        if 'questions' not in st.session_state:
            with st.spinner("Generating personalized questions for you..."):
                st.session_state.questions = system.conduct_screening(session)
        questions = st.session_state.questions
        if not questions:
            st.error("Unable to generate questions. Please try again.")
            if st.button("Restart"):
                st.session_state.clear()
                st.rerun()
            return
        st.markdown(f"**Age Group:** {session.age_group.value} years")
        st.markdown("Please answer the following questions honestly. Take your time.")
        responses = {}
        for i, question in enumerate(questions):
            st.markdown(f"**{i+1}. {question}**")
            response = st.text_area(
                f"Your response to question {i+1}:",
                key=f"q_{i}",
                height=100,
                placeholder="Share your thoughts here..."
            )
            if response.strip():
                responses[question] = response
        if len(responses) >= 10:
            if st.button("Submit Assessment", type="primary"):
                st.session_state.session = system.process_user_responses(session, responses)
                st.session_state.stage = 'analysis'
                st.rerun()
        else:
            st.warning(f"Please answer at least 10 questions to continue. ({len(responses)}/10 completed)")
    elif st.session_state.stage == 'analysis':
        st.header("Analyzing Your Responses...")
        with st.spinner("Processing your assessment and identifying patterns..."):
            session = system.analyze_and_classify(st.session_state.session)
            session = system.generate_bilan(session)
            st.session_state.session = session
        st.session_state.stage = 'results'
        st.rerun()
    elif st.session_state.stage == 'results':
        st.header("Your Mental Health Assessment Results")
        session = st.session_state.session
        risk_colors = {
            RiskLevel.LOW: "🟢",
            RiskLevel.MODERATE: "🟡",
            RiskLevel.HIGH: "🟠",
            RiskLevel.CRISIS: "🔴"
        }
        risk_labels = {
            RiskLevel.LOW: "Low Risk - You're doing well",
            RiskLevel.MODERATE: "Moderate Risk - Some concerns to address",
            RiskLevel.HIGH: "High Risk - Professional support recommended",
            RiskLevel.CRISIS: "Crisis Level - Immediate help needed"
        }
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"## {risk_colors.get(session.risk_level, '⚪')}")
        with col2:
            st.markdown(f"### {risk_labels.get(session.risk_level, 'Assessment Complete')}")
        if session.risk_level == RiskLevel.CRISIS:
            st.error(system.get_crisis_resources_display())
        st.markdown("### Your Mental Health Summary")
        if session.analysis:
            st.markdown(session.analysis)
        if session.recommendations:
            st.markdown("### Recommended Next Steps")
            for i, rec in enumerate(session.recommendations, 1):
                if session.risk_level == RiskLevel.CRISIS and i <= 2:
                    st.error(f"{i}. {rec}")
                elif session.risk_level == RiskLevel.HIGH and i == 1:
                    st.warning(f"{i}. {rec}")
                else:
                    st.info(f"{i}. {rec}")
        st.markdown("### What Would You Like to Do Next?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Take Personality Quiz", type="secondary"):
                st.info("Personality assessment feature coming soon!")
        with col2:
            if st.button("Get Personalized Resources", type="secondary"):
                st.info("Resource recommendation system coming soon!")
        with col3:
            if st.button("Start New Assessment", type="secondary"):
                if st.session_state.session:
                    system.complete_session(st.session_state.session.session_id)
                st.session_state.clear()
                st.rerun()
        with st.expander("Session Information"):
            summary = system.session_manager.get_session_summary(session.session_id)
            if summary:
                st.write(f"**Session ID:** {summary['session_id'][:8]}...")
                st.write(f"**Assessment Date:** {session.timestamp.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Age Group:** {summary['age_group']}")
                st.write(f"**Analysis Complete:** {'Yes' if summary['has_analysis'] else 'No'}")
            if st.button("Clear All Session Data", type="secondary"):
                system.complete_session(session.session_id)
                st.success("All session data has been securely cleared.")
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()