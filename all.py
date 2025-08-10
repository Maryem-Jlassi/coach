import os
import json
import uuid
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"
    API_BASE = "http://localhost:11434"
    SESSION_DURATION_MINUTES = 20

class AthleteLevel(Enum):
    BEGINNER = "Beginner (0-6 months)"
    INTERMEDIATE = "Intermediate (6 months - 2 years)"
    ADVANCED = "Advanced (2-5 years)"
    ELITE = "Elite (5+ years)"

class SportType(Enum):
    ENDURANCE = "Endurance Sports (Running, Cycling, Swimming)"
    STRENGTH = "Strength Training (Weightlifting, Powerlifting)"
    TEAM_SPORTS = "Team Sports (Football, Basketball, Soccer)"
    INDIVIDUAL = "Individual Sports (Tennis, Golf, Track)"
    COMBAT = "Combat Sports (Boxing, MMA, Wrestling)"
    MIXED = "Mixed/Cross Training"

@dataclass
class AthleteProfile:
    session_id: str
    name: str
    age: int
    weight: float
    height: float
    sport_type: SportType
    athlete_level: AthleteLevel
    training_frequency: int
    primary_goals: List[str] = field(default_factory=list)
    current_responses: Dict[str, str] = field(default_factory=dict)
    consultation_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class UnifiedMentalPerformanceCoach:
    """Unified AI Coach with integrated sub-agent capabilities"""
    
    def __init__(self):
        self.llm = LLM(model=Config.MODEL_NAME, base_url=Config.API_BASE)
        self.active_sessions = {}
        
    def create_unified_coach_agent(self, profile: AthleteProfile, consultation_phase: str) -> Agent:
        """Creates a unified coach agent that adapts based on consultation phase"""
        
        # Define coach personality based on athlete profile
        personality_traits = self._get_coach_personality(profile)
        
        return Agent(
            role=f'Unified Mental Performance & Wellness Coach - {consultation_phase.title()} Phase',
            goal=self._get_phase_goal(consultation_phase),
            backstory=f'''You are Coach Alex, a highly experienced mental performance coach 
            specializing in {profile.sport_type.value.lower()}. You have 15+ years of experience 
            working with athletes from {profile.athlete_level.value.lower()} to elite levels.
            
            Your coaching style is {personality_traits}. You understand that each athlete is 
            unique and requires personalized attention. You're known for your ability to:
            - Create genuine connections with athletes of all ages
            - Ask insightful questions that promote self-reflection
            - Provide practical, actionable advice
            - Maintain a supportive yet professional demeanor
            - Adapt your communication style to each individual
            
            Current athlete context:
            - {profile.name}, {profile.age} years old
            - {profile.sport_type.value} athlete
            - {profile.athlete_level.value} level
            - Trains {profile.training_frequency} times per week
            - Age-appropriate communication style: {self._get_age_communication_style(profile.age)}
            ''',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def _get_coach_personality(self, profile: AthleteProfile) -> str:
        """Generate personality traits based on athlete profile"""
        if profile.age <= 18:
            return "encouraging, patient, and supportive with a mentoring approach"
        elif profile.athlete_level in [AthleteLevel.BEGINNER, AthleteLevel.INTERMEDIATE]:
            return "motivational, educational, and confidence-building"
        else:
            return "direct, analytical, and performance-focused while remaining empathetic"
    
    def _get_age_communication_style(self, age: int) -> str:
        """Define communication style based on age"""
        if age <= 18:
            return "Friendly, encouraging, avoid overly clinical language"
        elif age <= 25:
            return "Motivational, goal-oriented, understanding of life balance challenges"
        elif age <= 40:
            return "Professional, time-conscious, focused on work-life-sport balance"
        else:
            return "Respectful, experience-acknowledging, health and longevity focused"
    
    def _get_phase_goal(self, phase: str) -> str:
        goals = {
            'welcome': 'Create rapport, confirm athlete profile, and set expectations for consultation',
            'question_generation': 'Generate personalized, dynamic questions based on athlete profile and current context',
            'assessment': 'Conduct thorough mental wellness assessment through natural conversation',
            'analysis': 'Analyze athlete responses and provide comprehensive coaching recommendations'
        }
        return goals.get(phase, 'Provide excellent mental performance coaching')
    
    def phase_1_welcome_and_confirmation(self, profile: AthleteProfile) -> str:
        """Phase 1: Welcome, rapport building, and profile confirmation"""
        agent = self.create_unified_coach_agent(profile, 'welcome')
        
        task = Task(
            description=f'''
            Welcome {profile.name} to their mental performance consultation. You are Coach Alex.
            
            Athlete Information:
            - Name: {profile.name}
            - Age: {profile.age} years old
            - Sport: {profile.sport_type.value}
            - Level: {profile.athlete_level.value}
            - Training: {profile.training_frequency} times per week
            - Height: {profile.height} cm, Weight: {profile.weight} kg
            
            Your welcome should:
            1. Greet them warmly by name and introduce yourself as Coach Alex
            2. Acknowledge their sport and experience level with respect
            3. Confirm their basic information is correct
            4. Explain the 20-minute consultation structure:
               - Welcome & goal setting (5 min)
               - Mental wellness assessment (10 min)
               - Analysis & personalized coaching plan (5 min)
            5. Ask about their current primary goals (1-3 specific goals)
            6. Inquire about what prompted them to seek mental performance coaching today
            7. Set a positive, supportive tone for the consultation
            8. Ensure they feel comfortable and ready to proceed
            
            Adapt your tone to be age-appropriate for a {profile.age}-year-old athlete.
            Be warm, professional, and genuinely interested in their success.
            End by asking if they're ready to move into the assessment phase.
            ''',
            expected_output='Warm welcome message with profile confirmation and consultation overview',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result)
    
    def phase_2_generate_personalized_questions(self, profile: AthleteProfile) -> List[str]:
        """Phase 2a: Generate dynamic, personalized questions"""
        agent = self.create_unified_coach_agent(profile, 'question_generation')
        
        # Create context for question generation
        context = self._build_question_context(profile)
        
        task = Task(
            description=f'''
            Generate 8 unique, personalized mental wellness questions for {profile.name}.
            
            Athlete Context:
            {context}
            
            Question Generation Guidelines:
            1. Each question must be unique and end with a question mark
            2. Questions should be conversational, not clinical
            3. Mix different aspects: current state, training mindset, goals, challenges, recovery
            4. Personalize based on their sport, level, and age
            5. Include 2-3 questions about recent training/performance
            6. Include 2-3 questions about mental state and stress
            7. Include 1-2 questions about goals and motivation
            8. Include 1-2 questions about lifestyle and recovery
            
            Question Types to Include:
            - Current mood and energy assessment
            - Recent training experience reflection
            - Goal clarity and motivation levels
            - Stress and pressure management
            - Recovery and lifestyle habits
            - Competition anxiety or confidence
            - Support system and external factors
            - Future aspirations and concerns
            
            Make questions feel like a caring coach having a natural conversation, 
            not like a clinical assessment. Use {profile.name}'s name occasionally.
            
            Return ONLY the 8 questions, numbered 1-8, nothing else.
            ''',
            expected_output='List of 8 personalized mental wellness questions',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        # Parse questions from result
        questions = self._parse_questions(str(result))
        return questions
    
    def _build_question_context(self, profile: AthleteProfile) -> str:
        """Build rich context for question generation"""
        context = f"""
        ATHLETE PROFILE:
        - Name: {profile.name}
        - Age: {profile.age} ({self._get_age_category(profile.age)})
        - Sport: {profile.sport_type.value}
        - Experience: {profile.athlete_level.value}
        - Training frequency: {profile.training_frequency} times per week
        - BMI category: {self._calculate_bmi_category(profile.weight, profile.height)}
        
        CONTEXTUAL FACTORS:
        - Time of year: {datetime.now().strftime('%B')} (consider seasonal training phases)
        - Age-specific considerations: {self._get_age_considerations(profile.age)}
        - Sport-specific mental aspects: {self._get_sport_mental_aspects(profile.sport_type)}
        - Experience level considerations: {self._get_level_considerations(profile.athlete_level)}
        """
        
        if profile.primary_goals:
            context += f"\n- Current goals: {', '.join(profile.primary_goals)}"
        
        return context
    
    def _get_age_category(self, age: int) -> str:
        if age <= 18: return "Youth athlete"
        elif age <= 25: return "Young adult athlete"
        elif age <= 40: return "Adult athlete"
        else: return "Masters athlete"
    
    def _calculate_bmi_category(self, weight: float, height: float) -> str:
        bmi = weight / ((height/100) ** 2)
        if bmi < 18.5: return "Underweight - may need nutrition focus"
        elif bmi < 25: return "Normal weight - good foundation"
        elif bmi < 30: return "Overweight - may impact performance"
        else: return "Obese - health considerations important"
    
    def _get_age_considerations(self, age: int) -> str:
        if age <= 18:
            return "School balance, parental pressure, identity development, peer relationships"
        elif age <= 25:
            return "Career decisions, independence, relationship changes, financial stress"
        elif age <= 40:
            return "Work-life balance, family responsibilities, time management, career peak"
        else:
            return "Health maintenance, injury prevention, legacy concerns, time limitations"
    
    def _get_sport_mental_aspects(self, sport_type: SportType) -> str:
        aspects = {
            SportType.ENDURANCE: "Mental toughness, pacing, dealing with discomfort, long-term focus",
            SportType.STRENGTH: "Confidence, body image, progressive overload mindset, injury fears",
            SportType.TEAM_SPORTS: "Communication, team dynamics, leadership, shared pressure",
            SportType.INDIVIDUAL: "Self-reliance, pressure management, perfectionism, isolation",
            SportType.COMBAT: "Aggression control, fear management, confidence, tactical thinking",
            SportType.MIXED: "Adaptability, varied motivation, scheduling challenges, goal clarity"
        }
        return aspects.get(sport_type, "General athletic mental performance")
    
    def _get_level_considerations(self, level: AthleteLevel) -> str:
        considerations = {
            AthleteLevel.BEGINNER: "Learning curve stress, comparison to others, habit formation",
            AthleteLevel.INTERMEDIATE: "Plateau frustration, goal refinement, technique focus",
            AthleteLevel.ADVANCED: "Performance pressure, specialization decisions, competition stress",
            AthleteLevel.ELITE: "Peak performance pressure, career longevity, public expectations"
        }
        return considerations.get(level, "General athletic development")
    
    def _parse_questions(self, result: str) -> List[str]:
        """Parse questions from LLM result"""
        questions = []
        lines = str(result).strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions
            if line and any(line.startswith(f"{i}.") for i in range(1, 11)):
                # Extract question after number and period
                if '.' in line:
                    question = line.split('.', 1)[1].strip()
                    if question.endswith('?'):
                        questions.append(question)
            # Also catch questions that end with ?
            elif line.endswith('?') and len(line.split()) > 3:
                questions.append(line)
        
        # Ensure we have exactly 8 questions
        if len(questions) >= 8:
            return questions[:8]
        elif len(questions) > 0:
            return questions
        else:
            # Fallback questions if parsing fails
            return self._get_fallback_questions(8)
    
    def _get_fallback_questions(self, num: int) -> List[str]:
        """Fallback questions if LLM generation fails"""
        fallback = [
            "How are you feeling about your athletic performance this week?",
            "What's been your biggest challenge in training recently?",
            "How confident do you feel heading into your next training session?",
            "What motivates you most in your sport right now?",
            "How well are you managing stress related to your athletic goals?",
            "What does your ideal training mindset look like?",
            "How do you typically prepare mentally before competing or training?",
            "What aspect of your mental game would you most like to improve?"
        ]
        return random.sample(fallback, min(num, len(fallback)))
    
    def phase_2_assessment_conversation(self, profile: AthleteProfile, questions: List[str]) -> str:
        """Phase 2b: Conduct natural conversation assessment"""
        agent = self.create_unified_coach_agent(profile, 'assessment')
        
        task = Task(
            description=f'''
            Conduct a natural, conversational mental wellness assessment with {profile.name}.
            
            You are Coach Alex having a supportive conversation. Use these questions as a guide,
            but adapt them naturally into flowing conversation:
            
            ASSESSMENT QUESTIONS TO EXPLORE:
            {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
            
            CONVERSATION APPROACH:
            1. Start by setting a comfortable, supportive tone
            2. Explain that you'll be asking questions to understand their mental game
            3. Make it feel like a natural coaching conversation, not an interrogation
            4. Use follow-up questions based on their responses
            5. Show empathy and understanding throughout
            6. Validate their feelings and experiences
            7. Use their name naturally in conversation
            8. Maintain professional warmth and genuine interest
            
            TONE AND STYLE:
            - Warm and approachable, like a trusted mentor
            - Professional but not clinical
            - Age-appropriate language for a {profile.age}-year-old
            - Sport-specific understanding of {profile.sport_type.value}
            - Encouraging and non-judgmental
            - Genuinely curious about their experience
            
            CONVERSATION FLOW:
            - Begin with easier, more general questions
            - Gradually move to deeper, more personal topics
            - Listen actively and respond to what they share
            - Build on their responses with thoughtful follow-ups
            - Create a safe space for honest reflection
            
            Remember: This is a conversation between a caring coach and their athlete,
            not a clinical assessment. Be human, warm, and genuinely invested in their success.
            ''',
            expected_output='Natural, flowing conversation that gathers deep insights about athlete mental state',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result)
    
    def phase_3_comprehensive_analysis_and_coaching(self, profile: AthleteProfile, 
                                                   assessment_conversation: str) -> str:
        """Phase 3: Comprehensive analysis and personalized coaching"""
        agent = self.create_unified_coach_agent(profile, 'analysis')
        
        # Prepare comprehensive data for analysis
        athlete_data = self._compile_athlete_data(profile, assessment_conversation)
        
        task = Task(
            description=f'''
            As Coach Alex, provide a comprehensive analysis and personalized coaching plan for {profile.name}.
            
            ATHLETE COMPREHENSIVE DATA:
            {athlete_data}
            
            ASSESSMENT CONVERSATION TRANSCRIPT:
            {assessment_conversation}
            
            COACHING ANALYSIS FRAMEWORK:
            
            1. MENTAL PERFORMANCE ASSESSMENT
            - Current mental state and readiness
            - Confidence levels and self-belief patterns
            - Stress and anxiety indicators
            - Motivation and drive assessment
            - Mental resilience and coping strategies
            - Focus and concentration abilities
            
            2. PERSONALIZED COACHING INSIGHTS
            Based on their responses, identify:
            - Key mental strengths to leverage
            - Areas needing development or attention
            - Potential mental performance barriers
            - Opportunities for growth and improvement
            - Risk factors for burnout or mental fatigue
            
            3. CUSTOMIZED MENTAL TRAINING PLAN
            Provide specific, actionable recommendations:
            
            A) DAILY MENTAL PRACTICES (personalized for {profile.sport_type.value})
            - Morning mental preparation routines
            - Pre-training visualization techniques
            - Post-training reflection practices
            - Evening wind-down and recovery methods
            
            B) PERFORMANCE ENHANCEMENT STRATEGIES
            - Competition day mental preparation
            - Pressure management techniques
            - Confidence building exercises
            - Focus and concentration improvement
            - Stress reduction methods
            
            C) LIFESTYLE OPTIMIZATION
            - Sleep hygiene for mental recovery
            - Nutrition for cognitive performance
            - Stress management outside sport
            - Work-life-sport balance strategies
            - Social support system strengthening
            
            4. MOTIVATIONAL COACHING PLAN
            - Personalized affirmations based on their goals
            - Achievement recognition and celebration methods
            - Progress tracking and milestone setting
            - Overcoming setbacks and maintaining resilience
            - Long-term motivation sustainability
            
            5. IMMEDIATE ACTION STEPS
            - What to implement in the next 24 hours
            - Weekly practices to establish
            - Monthly check-in goals
            - Warning signs to monitor
            
            COACHING DELIVERY STYLE:
            - Maintain Coach Alex's warm, professional tone
            - Use their name and reference specific things they shared
            - Be encouraging while being realistic
            - Provide specific, actionable advice
            - Show confidence in their ability to improve
            - End with inspiration and next steps
            
            Make this feel like a comprehensive coaching session summary that an athlete 
            would receive from their personal mental performance coach.
            ''',
            expected_output='Comprehensive mental performance analysis with personalized coaching plan and actionable recommendations',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        return str(result)
    
    def _compile_athlete_data(self, profile: AthleteProfile, conversation: str) -> str:
        """Compile all athlete data for comprehensive analysis"""
        return f'''
        ATHLETE PROFILE SUMMARY:
        - Name: {profile.name}
        - Age: {profile.age} years ({self._get_age_category(profile.age)})
        - Physical: {profile.height}cm, {profile.weight}kg
        - Sport: {profile.sport_type.value}
        - Experience: {profile.athlete_level.value}
        - Training: {profile.training_frequency} sessions/week
        - Session Date: {profile.timestamp.strftime('%B %d, %Y')}
        - Goals: {', '.join(profile.primary_goals) if profile.primary_goals else 'To be discussed'}
        
        CONTEXTUAL FACTORS:
        - Age considerations: {self._get_age_considerations(profile.age)}
        - Sport mental aspects: {self._get_sport_mental_aspects(profile.sport_type)}
        - Experience level factors: {self._get_level_considerations(profile.athlete_level)}
        
        USER RESPONSES FROM ASSESSMENT:
        {chr(10).join([f"Response to '{q}': {r}" for q, r in profile.current_responses.items()])}
        '''
    
    def _get_age_category(self, age: int) -> str:
        if age <= 18: return "Youth athlete"
        elif age <= 25: return "Young adult athlete" 
        elif age <= 40: return "Adult athlete"
        else: return "Masters athlete"
    
    def _get_age_considerations(self, age: int) -> str:
        if age <= 18:
            return "Academic pressure, social development, parental expectations, identity formation"
        elif age <= 25:
            return "Career establishment, independence, relationship development, financial concerns"
        elif age <= 40:
            return "Career demands, family responsibilities, time constraints, peak performance window"
        else:
            return "Health maintenance, injury prevention, wisdom sharing, legacy building"
    
    def _get_sport_mental_aspects(self, sport_type: SportType) -> str:
        aspects = {
            SportType.ENDURANCE: "Mental toughness, pain tolerance, pacing strategy, monotony management",
            SportType.STRENGTH: "Confidence building, progressive mindset, body image, fear of injury",
            SportType.TEAM_SPORTS: "Communication skills, team dynamics, shared accountability, role clarity",
            SportType.INDIVIDUAL: "Self-reliance, pressure handling, perfectionism management, isolation coping",
            SportType.COMBAT: "Aggression control, fear management, tactical thinking, confidence under pressure",
            SportType.MIXED: "Adaptability, goal prioritization, scheduling balance, diverse skill confidence"
        }
        return aspects.get(sport_type, "General athletic mental performance factors")
    
    def _get_level_considerations(self, level: AthleteLevel) -> str:
        considerations = {
            AthleteLevel.BEGINNER: "Learning curve management, expectation setting, habit formation, comparison avoidance",
            AthleteLevel.INTERMEDIATE: "Plateau navigation, goal refinement, technique focus, progress patience",
            AthleteLevel.ADVANCED: "Performance optimization, competition strategy, specialization decisions, pressure handling",
            AthleteLevel.ELITE: "Peak performance maintenance, career sustainability, public pressure, legacy concerns"
        }
        return considerations.get(level, "General development considerations")
    
    def conduct_full_consultation(self, profile: AthleteProfile, responses: Dict[str, str]) -> Dict[str, str]:
        """Conduct complete consultation with all phases"""
        profile.current_responses = responses
        self.active_sessions[profile.session_id] = profile
        
        # Phase 1: Welcome and confirmation
        welcome_result = self.phase_1_welcome_and_confirmation(profile)
        
        # Phase 2a: Generate personalized questions
        questions = self.phase_2_generate_personalized_questions(profile)
        
        # Phase 2b: Assessment conversation
        assessment_result = self.phase_2_assessment_conversation(profile, questions)
        
        # Phase 3: Analysis and coaching
        coaching_result = self.phase_3_comprehensive_analysis_and_coaching(profile, assessment_result)
        
        return {
            'welcome': welcome_result,
            'assessment': assessment_result,
            'coaching': coaching_result,
            'questions_used': questions
        }

# Streamlit UI Implementation
def main():
    st.set_page_config(
        page_title="Mental Performance & Wellness Coach", 
        page_icon="🧠💪", 
        layout="wide"
    )
    
    st.title("🧠💪 Mental Performance & Wellness Coach")
    st.markdown("*Your Personal AI Sports Psychology Coach - Dynamic 20-minute consultation*")
    
    # Initialize system
    if 'coach_system' not in st.session_state:
        st.session_state.coach_system = UnifiedMentalPerformanceCoach()
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = None
    
    if 'consultation_stage' not in st.session_state:
        st.session_state.consultation_stage = 'setup'
    
    system = st.session_state.coach_system
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Today's Consultation")
        st.success("""
        **Your Coach: Alex**
        
        20-minute personalized session:
        • Welcome & Goal Setting (5 min)
        • Dynamic Mental Assessment (10 min) 
        • Analysis & Coaching Plan (5 min)
        """)
        
        st.header("🌟 What Makes This Special")
        st.markdown("""
        ✨ **Dynamic Questions** - Never the same twice
        
        🎯 **Personalized Approach** - Adapted to your sport, age, and level
        
        🤝 **Conversational Style** - Natural coaching dialogue
        
        📈 **Actionable Results** - Specific mental training plan
        """)
        
        if st.session_state.current_profile:
            st.header("📊 Current Session")
            profile = st.session_state.current_profile
            st.write(f"**Athlete:** {profile.name}")
            st.write(f"**Sport:** {profile.sport_type.value}")
            st.write(f"**Level:** {profile.athlete_level.value}")
            st.write(f"**Started:** {profile.timestamp.strftime('%H:%M')}")
    
    # Main consultation flow
    if st.session_state.consultation_stage == 'setup':
        show_athlete_profile_setup()
    elif st.session_state.consultation_stage == 'welcome':
        show_welcome_phase()
    elif st.session_state.consultation_stage == 'assessment':
        show_dynamic_assessment_phase()
    elif st.session_state.consultation_stage == 'coaching':
        show_coaching_results_phase()

def show_athlete_profile_setup():
    st.header("🏃‍♂️ Welcome to Your Personal Coaching Session")
    st.markdown("""
    Hi there! I'm **Coach Alex**, your AI mental performance specialist. I'm here to help you 
    optimize your mental game and overall wellness as an athlete. Let's start by getting to know you better.
    """)
    
    with st.form("athlete_profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            name = st.text_input("First Name", placeholder="What should I call you?")
            age = st.number_input("Age", min_value=13, max_value=80, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            height = st.number_input("Height (cm)", min_value=120, max_value=230, value=170)
        
        with col2:
            st.subheader("Athletic Profile")
            sport_type = st.selectbox(
                "Primary Sport Category",
                options=list(SportType),
                format_func=lambda x: x.value
            )
            
            athlete_level = st.selectbox(
                "Experience Level",
                options=list(AthleteLevel),
                format_func=lambda x: x.value
            )
            
            training_frequency = st.number_input(
                "Training sessions per week",
                min_value=1, max_value=14, value=4
            )
        
        st.subheader("What brings you here today?")
        goals_text = st.text_area(
            "Share 1-3 goals you'd like to work on in this consultation:",
            placeholder="e.g., improve pre-competition confidence, manage training stress, enhance focus...",
            height=100
        )
        
        submitted = st.form_submit_button("Start My Consultation with Coach Alex", type="primary")
        
        if submitted and name:
            goals = [goal.strip() for goal in goals_text.split(',') if goal.strip()] if goals_text else []
            
            profile = AthleteProfile(
                session_id=str(uuid.uuid4()),
                name=name,
                age=age,
                weight=weight,
                height=height,
                sport_type=sport_type,
                athlete_level=athlete_level,
                training_frequency=training_frequency,
                primary_goals=goals
            )
            
            st.session_state.current_profile = profile
            st.session_state.consultation_stage = 'welcome'
            st.rerun()
        elif submitted:
            st.error("Please enter your name to continue.")

def show_welcome_phase():
    st.header("👋 Phase 1: Welcome & Profile Confirmation")
    profile = st.session_state.current_profile
    system = st.session_state.coach_system
    
    if 'welcome_complete' not in st.session_state:
        with st.spinner("Coach Alex is reviewing your profile and preparing your personalized consultation..."):
            welcome_result = system.phase_1_welcome_and_confirmation(profile)
            st.session_state.welcome_result = welcome_result
            st.session_state.welcome_complete = True
    
    st.success("🎤 Coach Alex:")
    st.write(st.session_state.welcome_result)
    
    st.markdown("---")
    if st.button("I'm ready to begin the assessment!", type="primary", use_container_width=True):
        st.session_state.consultation_stage = 'assessment'
        st.rerun()

def show_dynamic_assessment_phase():
    st.header("🧠 Phase 2: Dynamic Mental Assessment")
    st.markdown("*Coach Alex will ask you personalized questions based on your profile*")
    
    profile = st.session_state.current_profile
    system = st.session_state.coach_system
    
    # Generate dynamic questions
    if 'dynamic_questions' not in st.session_state:
        with st.spinner("Coach Alex is creating personalized questions just for you..."):
            questions = system.phase_2_generate_personalized_questions(profile)
            st.session_state.dynamic_questions = questions
    
    questions = st.session_state.dynamic_questions
    
    if 'assessment_conversation' not in st.session_state:
        with st.spinner("Coach Alex is preparing the assessment conversation..."):
            conversation = system.phase_2_assessment_conversation(profile, questions)
            st.session_state.assessment_conversation = conversation
    
    st.success("🎤 Coach Alex - Assessment Conversation:")
    st.write(st.session_state.assessment_conversation)
    
    st.markdown("---")
    st.subheader("📝 Your Responses")
    st.markdown("*Please respond to the questions Coach Alex discussed with you*")
    
    responses = {}
    for i, question in enumerate(questions):
        response = st.text_area(
            f"**{question}**",
            key=f"dynamic_response_{i}",
            height=100,
            placeholder="Share your honest thoughts and feelings..."
        )
        if response.strip():
            responses[question] = response
    
    # Progress indicator
    progress = len(responses) / len(questions)
    st.progress(progress, text=f"Assessment Progress: {len(responses)}/{len(questions)} questions answered")
    
    if len(responses) >= 6:
        if st.button("Complete Assessment & Get My Coaching Plan", type="primary", use_container_width=True):
            profile.current_responses = responses
            system.update_session(profile)
            st.session_state.consultation_stage = 'coaching'
            st.rerun()
    else:
        st.warning(f"Please answer at least 6 questions to get meaningful coaching insights ({len(responses)}/{len(questions)} completed)")

def show_coaching_results_phase():
    st.header("🏆 Phase 3: Your Personalized Coaching Plan")
    
    profile = st.session_state.current_profile
    system = st.session_state.coach_system
    
    if 'coaching_analysis' not in st.session_state:
        with st.spinner("Coach Alex is analyzing your responses and creating your personalized mental performance plan..."):
            assessment_data = st.session_state.get('assessment_conversation', '')
            coaching_result = system.phase_3_comprehensive_analysis_and_coaching(profile, assessment_data)
            st.session_state.coaching_analysis = coaching_result
    
    st.success("🎤 Coach Alex - Your Complete Coaching Analysis:")
    st.markdown(st.session_state.coaching_analysis)
    
    st.markdown("---")
    
    # Action buttons
    st.subheader("🎯 What's Next?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Get Daily Reminders", type="secondary", use_container_width=True):
            show_daily_reminders(profile)
    
    with col2:
        if st.button("📋 Save My Plan", type="secondary", use_container_width=True):
            show_save_options(profile)
    
    with col3:
        if st.button("🔄 New Consultation", type="secondary", use_container_width=True):
            start_new_consultation()
    
    # Session summary
    with st.expander("📊 Consultation Summary"):
        st.write(f"**Athlete:** {profile.name}")
        st.write(f"**Session ID:** {profile.session_id[:8]}...")
        st.write(f"**Date:** {profile.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Sport Focus:** {profile.sport_type.value}")
        st.write(f"**Questions Answered:** {len(profile.current_responses)}")
        st.write(f"**Goals Discussed:** {', '.join(profile.primary_goals) if profile.primary_goals else 'General performance improvement'}")

def show_daily_reminders(profile: AthleteProfile):
    """Show daily reminder suggestions"""
    st.info(f"""
    📱 **Daily Mental Training Reminders for {profile.name}:**
    
    🌅 **Morning (5 min):**
    - Review your daily intention
    - Quick visualization of successful training
    - Positive affirmation practice
    
    🏃‍♂️ **Pre-Training (3 min):**
    - Breathing exercise for focus
    - Mental preparation routine
    - Goal setting for the session
    
    🌙 **Evening (5 min):**
    - Reflect on the day's training
    - Gratitude practice
    - Tomorrow's mental preparation
    
    *Tip: Set phone reminders for these times to build consistent mental training habits!*
    """)

def show_save_options(profile: AthleteProfile):
    """Show options for saving the coaching plan"""
    st.info(f"""
    📋 **Save Your Coaching Plan, {profile.name}:**
    
    ✅ **What to Save:**
    - Complete mental performance analysis
    - Personalized daily practices
    - Competition preparation routines
    - Stress management strategies
    - Progress tracking methods
    
    💡 **Recommended Actions:**
    1. Screenshot or copy the coaching analysis above
    2. Create a note in your phone with key daily practices
    3. Share relevant insights with your training coach
    4. Schedule weekly self-check-ins using the provided framework
    
    🔄 **Return for Updates:**
    Come back monthly for consultation updates as your training evolves!
    """)

def start_new_consultation():
    """Start a new consultation session"""
    # Clear all session state
    for key in list(st.session_state.keys()):
        if key not in ['coach_system']:  # Keep the system initialized
            del st.session_state[key]
    
    st.session_state.consultation_stage = 'setup'
    st.rerun()

# Helper classes integration
class SessionManager:
    """Manages consultation sessions"""
    
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, profile: AthleteProfile) -> str:
        self.sessions[profile.session_id] = {
            'profile': profile,
            'created_at': datetime.now(),
            'status': 'active'
        }
        return profile.session_id
    
    def get_session(self, session_id: str) -> Optional[AthleteProfile]:
        session_data = self.sessions.get(session_id)
        return session_data['profile'] if session_data else None
    
    def update_session(self, profile: AthleteProfile):
        if profile.session_id in self.sessions:
            self.sessions[profile.session_id]['profile'] = profile
            self.sessions[profile.session_id]['last_updated'] = datetime.now()
    
    def complete_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]['status'] = 'completed'
            self.sessions[session_id]['completed_at'] = datetime.now()

# Enhanced unified coach with session management
class UnifiedMentalPerformanceCoach(UnifiedMentalPerformanceCoach):
    """Enhanced coach with session management"""
    
    def __init__(self):
        super().__init__()
        self.session_manager = SessionManager()
    
    def update_session(self, profile: AthleteProfile):
        """Update session in both systems"""
        self.active_sessions[profile.session_id] = profile
        self.session_manager.update_session(profile)

# Run the application
if __name__ == "__main__":
    # Display startup information
    st.markdown("---")
    with st.expander("🔧 System Requirements", expanded=False):
        st.markdown("""
        **Before starting, ensure you have:**
        
        1. **Ollama installed and running:**
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull the model
        ollama pull llama3.2:latest
        
        # Start Ollama server
        ollama serve
        ```
        
        2. **Required Python packages:**
        ```bash
        pip install streamlit crewai
        ```
        
        3. **Ollama running on localhost:11434** (default port)
        
        **Ready?** Complete your athlete profile above to begin your personalized consultation!
        """)
    
    main()