import os
import json
import uuid
import random
import sqlite3
import asyncio
import tempfile
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
import threading
import time

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"
    API_BASE = "http://localhost:11434"
    SESSION_DURATION_MINUTES = 20
    TTS_LANGUAGE = "en"
    TTS_SLOW = False
    VOICE_RATE = 160  # Words per minute for pyttsx3
    DATABASE_PATH = "coach_sessions.db"

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
    consultation_history: List[Dict] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class VoiceService:
    """Open-source TTS and Speech Recognition service"""
    
    def __init__(self):
        self.tts_engine = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.setup_tts()
        
    def setup_tts(self):
        """Initialize TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a good voice (prefer female voices for warmth)
            for voice in voices:
                if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            # Set speech rate for natural conversation
            self.tts_engine.setProperty('rate', Config.VOICE_RATE)
            self.tts_engine.setProperty('volume', 0.9)
            
        except Exception as e:
            st.error(f"TTS initialization error: {e}")
            self.tts_engine = None
    
    def text_to_speech_gtts(self, text: str) -> bytes:
        """Generate speech using Google TTS (offline alternative)"""
        try:
            # Clean text for better speech
            clean_text = self._prepare_text_for_speech(text)
            
            tts = gTTS(text=clean_text, lang=Config.TTS_LANGUAGE, slow=Config.TTS_SLOW)
            
            # Save to BytesIO buffer
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None
    
    def text_to_speech_pyttsx3(self, text: str, save_path: str = None) -> str:
        """Generate speech using pyttsx3 (fully offline)"""
        if not self.tts_engine:
            return None
            
        try:
            clean_text = self._prepare_text_for_speech(text)
            
            if save_path:
                self.tts_engine.save_to_file(clean_text, save_path)
                self.tts_engine.runAndWait()
                return save_path
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_path = temp_file.name
                temp_file.close()
                
                self.tts_engine.save_to_file(clean_text, temp_path)
                self.tts_engine.runAndWait()
                
                return temp_path
                
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """Prepare text for more natural speech"""
        # Add natural pauses
        text = text.replace('. ', '... ')
        text = text.replace('!', '!')
        text = text.replace('?', '?')
        
        # Handle common abbreviations
        text = text.replace('e.g.', 'for example')
        text = text.replace('i.e.', 'that is')
        text = text.replace('etc.', 'and so on')
        
        # Add emphasis for key words
        text = text.replace('important', 'really important')
        text = text.replace('great', 'really great')
        
        return text
    
    def listen_for_speech(self, timeout: int = 10) -> Optional[str]:
        """Listen for speech input using microphone"""
        try:
            with self.microphone as source:
                st.write("🎤 Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=30)
                
            st.write("🔄 Processing your response...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            st.warning("⏱️ No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("🤔 Couldn't understand the audio. Please speak clearly and try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            return None

class DatabaseManager:
    """Manages athlete session persistence"""
    
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for session storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS athlete_sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                weight REAL,
                height REAL,
                sport_type TEXT,
                athlete_level TEXT,
                training_frequency INTEGER,
                primary_goals TEXT,
                current_responses TEXT,
                consultation_history TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                phase TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES athlete_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_athlete_profile(self, profile: AthleteProfile):
        """Save or update athlete profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO athlete_sessions 
            (session_id, name, age, weight, height, sport_type, athlete_level, 
             training_frequency, primary_goals, current_responses, consultation_history, 
             preferences, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.session_id,
            profile.name,
            profile.age,
            profile.weight,
            profile.height,
            profile.sport_type.value,
            profile.athlete_level.value,
            profile.training_frequency,
            json.dumps(profile.primary_goals),
            json.dumps(profile.current_responses),
            json.dumps(profile.consultation_history),
            json.dumps(profile.preferences),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
    
    def load_athlete_profile(self, session_id: str) -> Optional[AthleteProfile]:
        """Load athlete profile by session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM athlete_sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return AthleteProfile(
                session_id=row[0],
                name=row[1],
                age=row[2],
                weight=row[3],
                height=row[4],
                sport_type=SportType(row[5]),
                athlete_level=AthleteLevel(row[6]),
                training_frequency=row[7],
                primary_goals=json.loads(row[8]),
                current_responses=json.loads(row[9]),
                consultation_history=json.loads(row[10]),
                preferences=json.loads(row[11])
            )
        return None
    
    def get_athlete_history(self, name: str) -> List[Dict]:
        """Get consultation history for an athlete by name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, created_at, sport_type, primary_goals 
            FROM athlete_sessions 
            WHERE name = ? 
            ORDER BY created_at DESC
        ''', (name,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': row[0],
                'date': row[1],
                'sport': row[2],
                'goals': json.loads(row[3])
            }
            for row in rows
        ]
    
    def log_consultation_phase(self, session_id: str, phase: str, content: str):
        """Log consultation phase for debugging and improvement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consultation_logs (session_id, phase, content)
            VALUES (?, ?, ?)
        ''', (session_id, phase, content))
        
        conn.commit()
        conn.close()

class VoiceEnabledCoach:
    """Voice-first mental performance coach with human-like interaction"""
    
    def __init__(self):
        self.llm = LLM(model=Config.MODEL_NAME, base_url=Config.API_BASE)
        self.voice_service = VoiceService()
        self.db_manager = DatabaseManager()
        self.active_sessions = {}
        pygame.mixer.init()  # Initialize pygame for audio playback
        
    def create_human_coach_agent(self, profile: AthleteProfile, phase: str) -> Agent:
        """Creates a highly human-like coach agent"""
        
        history = self.db_manager.get_athlete_history(profile.name)
        history_context = self._build_history_context(history, profile) if len(history) > 1 else ""
        
        personality = self._get_human_personality(profile)
        
        return Agent(
            role=f'Coach Alex - Voice-Interactive Mental Performance Coach',
            goal=self._get_phase_goal(phase),
            backstory=f'''You are Coach Alex, a warm, caring, and highly experienced mental performance coach. 
            You speak naturally and conversationally, as if you're having a face-to-face conversation 
            with an athlete you genuinely care about.
            
            YOUR HUMAN QUALITIES:
            - Naturally warm and encouraging voice
            - Remember personal details and reference them
            - Use conversational speech patterns ("Well, you know..." "That's really interesting...")
            - Show genuine emotional responses to what athletes share
            - Use natural pauses and emphasis in speech
            - Occasionally use gentle humor when appropriate
            - Express empathy authentically ("I can really understand that...")
            - Use the athlete's name naturally, not excessively
            
            YOUR COACHING EXPERTISE:
            - 20+ years working with athletes at all levels
            - Specialized in {profile.sport_type.value.lower()}
            - Expert in mental performance and sports psychology
            - Known for making athletes feel completely comfortable
            - Excellent at asking questions that promote self-reflection
            - Skilled at providing actionable, practical advice
            
            CURRENT ATHLETE:
            - {profile.name}, {profile.age} years old
            - {profile.sport_type.value} at {profile.athlete_level.value} level
            - Trains {profile.training_frequency} times per week
            - Primary goals: {', '.join(profile.primary_goals) if profile.primary_goals else 'To be discovered'}
            
            {history_context}
            
            SPEECH STYLE FOR VOICE DELIVERY:
            - {personality['speech_style']}
            - {personality['energy_level']}
            - {personality['supportiveness']}
            
            IMPORTANT: Since this will be converted to speech, use natural speaking patterns:
            - Include conversational connectors ("So...", "Now...", "You know...")
            - Use contractions naturally ("I'll", "you're", "that's")
            - Vary sentence length for natural rhythm
            - Include empathetic responses ("Mm-hmm", "I see", "That makes sense")
            - Use encouraging phrases that sound genuine
            - Speak as if you're truly interested in their success
            ''',
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def _get_human_personality(self, profile: AthleteProfile) -> Dict[str, str]:
        """Get human-like personality traits for voice interaction"""
        if profile.age <= 18:
            return {
                'speech_style': 'Patient and encouraging, like a supportive mentor',
                'energy_level': 'Enthusiastic but not overwhelming, builds confidence',
                'supportiveness': 'Very nurturing, celebrates small wins, gentle with challenges'
            }
        elif profile.age <= 25:
            return {
                'speech_style': 'Energetic and motivational, speaks like a peer-mentor',
                'energy_level': 'High energy, matches their drive and ambition',
                'supportiveness': 'Goal-oriented support, understands life pressures'
            }
        elif profile.age <= 40:
            return {
                'speech_style': 'Professional yet warm, respects their experience',
                'energy_level': 'Focused and efficient, values their time',
                'supportiveness': 'Direct but caring, practical advice-focused'
            }
        else:
            return {
                'speech_style': 'Respectful and wise, acknowledges their experience',
                'energy_level': 'Calm and steady, focuses on sustainability',
                'supportiveness': 'Health-conscious, emphasizes wisdom and longevity'
            }
    
    def _build_history_context(self, history: List[Dict], profile: AthleteProfile) -> str:
        """Build context from previous consultations"""
        if not history:
            return ""
        
        context = f"\nRELATIONSHIP HISTORY:\n"
        context += f"You've worked with {profile.name} before! You have {len(history)} previous sessions together.\n"
        
        for i, session in enumerate(history[:2]):  # Last 2 sessions
            date = datetime.fromisoformat(session['date']).strftime('%B %d')
            context += f"- {date}: Worked on {', '.join(session['goals'][:2])}\n"
        
        context += f"\nRemember to acknowledge your ongoing relationship and reference past work naturally."
        return context
    
    def _get_phase_goal(self, phase: str) -> str:
        goals = {
            'welcome': 'Create genuine human connection and rapport through natural conversation',
            'questions': 'Ask personalized questions that feel like natural curiosity from a caring coach',
            'assessment': 'Conduct empathetic voice conversation that feels like talking to a trusted mentor',
            'coaching': 'Provide insightful, caring coaching advice that motivates and inspires'
        }
        return goals.get(phase, 'Provide exceptional human-like coaching experience')
    
    def phase_1_voice_welcome(self, profile: AthleteProfile) -> Dict[str, Any]:
        """Phase 1: Voice-delivered welcome with human warmth"""
        agent = self.create_human_coach_agent(profile, 'welcome')
        
        task = Task(
            description=f'''
            You are Coach Alex meeting {profile.name} for a voice-based mental performance consultation.
            This is like a phone call or video chat with a real coach. Be naturally conversational and warm.
            
            ATHLETE INFO:
            - Name: {profile.name}
            - Age: {profile.age} years old
            - Sport: {profile.sport_type.value}
            - Level: {profile.athlete_level.value}
            - Training: {profile.training_frequency} times per week
            - Goals: {', '.join(profile.primary_goals) if profile.primary_goals else 'General improvement'}
            
            VOICE WELCOME SCRIPT:
            Create a natural, conversational welcome that includes:
            
            1. WARM GREETING (sound genuinely happy to meet them):
            "Hi {profile.name}! This is Coach Alex. It's so wonderful to meet you!"
            
            2. IMMEDIATE CONNECTION (show interest in them personally):
            - Acknowledge their sport with genuine enthusiasm
            - Comment positively on their commitment to training
            - Make them feel special for seeking mental performance help
            
            3. NATURAL CONVERSATION FLOW:
            - "I'm really excited to spend the next 20 minutes with you"
            - "Before we dive in, how are you feeling today?"
            - Reference their goals naturally
            - Ask about their current energy and mindset
            
            4. CONSULTATION EXPLANATION (keep it conversational):
            - Explain this is interactive and voice-based
            - "We'll chat for about 20 minutes total"
            - "I'll ask you some questions to understand your mental game"
            - "Then I'll share some personalized strategies just for you"
            
            5. COMFORT AND READINESS:
            - Make sure they're in a good place to talk
            - "Take your time with answers, there's no pressure"
            - "I'm here to support you, not judge anything"
            - End with: "Are you ready to dive in? I'm really looking forward to our conversation!"
            
            VOICE DELIVERY STYLE:
            - Speak like you're genuinely excited to help them
            - Use natural speech patterns and rhythm
            - Include warm laughter or enthusiasm where appropriate
            - Sound like a real person, not a robot
            - Use their name 2-3 times naturally
            - Show authentic interest in their athletic journey
            - Keep energy positive but not overwhelming
            
            Remember: This will be spoken aloud, so make it sound like natural human speech!
            ''',
            expected_output='Natural, warm voice welcome that builds genuine human connection',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        # Generate voice audio
        voice_audio = self.generate_voice_audio(str(result))
        
        # Log welcome phase
        self.db_manager.log_consultation_phase(profile.session_id, 'voice_welcome', str(result))
        
        return {
            'text': str(result),
            'audio_file': voice_audio,
            'phase': 'welcome_complete'
        }
    
    def phase_2_voice_questions_generation(self, profile: AthleteProfile) -> List[Dict[str, str]]:
        """Phase 2a: Generate individual voice questions with personalized introductions"""
        agent = self.create_human_coach_agent(profile, 'questions')
        
        context = self._build_comprehensive_context(profile)
        
        task = Task(
            description=f'''
            As Coach Alex, generate 6 individual voice questions for {profile.name}.
            Each question should have a natural introduction and the actual question.
            
            ATHLETE CONTEXT:
            {context}
            
            VOICE QUESTION GENERATION:
            For each question, provide:
            1. Natural introduction/setup (1-2 sentences)
            2. The actual question
            3. Encouraging prompt to take their time
            
            Example format:
            "So {profile.name}, I'd love to understand your current mindset better. How are you feeling about your training this week? Take your time, I'm really interested in hearing your thoughts."
            
            QUESTION CATEGORIES (generate 1 from each + 2 bonus):
            1. Current mental state and energy
            2. Recent training experiences and confidence
            3. Goal clarity and motivation
            4. Stress and pressure management
            5. Recovery and lifestyle balance
            6. Support system and relationships
            
            VOICE-OPTIMIZED REQUIREMENTS:
            - Make each question feel like genuine coaching curiosity
            - Use natural speech patterns and conversational flow
            - Include their name appropriately (not in every question)
            - Add empathetic setup phrases
            - Make questions open-ended and reflective
            - Use sport-specific language naturally
            - Sound genuinely interested, not clinical
            - Include transition phrases between setup and question
            
            PERSONALIZATION FACTORS:
            - Age-appropriate language for {profile.age} year old
            - {profile.sport_type.value} specific considerations
            - {profile.athlete_level.value} experience level
            - Current season: {datetime.now().strftime('%B')}
            - Training load: {profile.training_frequency} sessions/week
            
            Generate exactly 6 complete voice questions with natural introductions.
            Each should be ready to be spoken aloud by Coach Alex.
            ''',
            expected_output='6 complete voice questions with natural introductions for spoken delivery',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        # Parse individual questions with their introductions
        questions = self._parse_voice_questions(str(result))
        
        # Generate audio for each question
        voice_questions = []
        for i, question in enumerate(questions):
            audio_file = self.generate_voice_audio(question, f"question_{i+1}")
            voice_questions.append({
                'text': question,
                'audio_file': audio_file,
                'number': i + 1
            })
        
        # Log questions
        self.db_manager.log_consultation_phase(
            profile.session_id, 
            'voice_questions_generated', 
            json.dumps([q['text'] for q in voice_questions])
        )
        
        return voice_questions
    
    def _build_comprehensive_context(self, profile: AthleteProfile) -> str:
        """Build comprehensive context for question generation"""
        return f"""
        COMPLETE ATHLETE CONTEXT:
        
        Personal:
        - {profile.name}, {profile.age} years old ({self._get_age_category(profile.age)})
        - {self._get_age_considerations(profile.age)}
        
        Athletic:
        - Sport: {profile.sport_type.value}
        - Experience: {profile.athlete_level.value}
        - Training: {profile.training_frequency} sessions/week
        - Mental aspects: {self._get_sport_mental_aspects(profile.sport_type)}
        
        Current Context:
        - Date: {datetime.now().strftime('%A, %B %d, %Y')}
        - Season: {self._get_current_season()}
        - Training load assessment: {self._assess_training_load(profile.training_frequency)}
        
        Goals:
        {', '.join(profile.primary_goals) if profile.primary_goals else 'To be discovered through conversation'}
        """
    
    def _parse_voice_questions(self, result: str) -> List[str]:
        """Parse voice questions with their natural introductions"""
        questions = []
        text = str(result).strip()
        
        # Split by common patterns
        potential_questions = []
        
        # Method 1: Split by numbered patterns
        lines = text.split('\n')
        current_question = ""
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                if current_question:
                    potential_questions.append(current_question.strip())
                current_question = line.split('.', 1)[1].strip() if '.' in line else line
            elif line and current_question:
                current_question += " " + line
        
        if current_question:
            potential_questions.append(current_question.strip())
        
        # Method 2: Look for question marks
        if not potential_questions:
            sentences = text.replace('\n', ' ').split('.')
            for sentence in sentences:
                if '?' in sentence:
                    potential_questions.append(sentence.strip() + '.')
        
        # Clean and validate questions
        for q in potential_questions:
            if len(q.split()) > 10 and ('?' in q or q.endswith('.')):
                questions.append(q)
        
        # Fallback questions if parsing fails
        if len(questions) < 4:
            questions.extend(self._get_voice_fallback_questions(profile))
        
        return questions[:6]
    
    def _get_voice_fallback_questions(self, profile: AthleteProfile) -> List[str]:
        """Voice-optimized fallback questions"""
        fallbacks = [
            f"So {profile.name}, I'd love to start by understanding how you're feeling right now. What's your current energy level like, and how are you feeling about your training lately?",
            f"I'm really curious about your recent experiences. Can you tell me about a training session this week that stood out to you, either positively or as a challenge?",
            f"Let's talk about confidence for a moment. When you think about your {profile.sport_type.value.lower()}, how confident are you feeling these days, and what affects that confidence most?",
            f"Now, {profile.name}, stress is something every athlete deals with. How do you typically handle pressure, whether it's from training, competition, or just life in general?",
            f"I'm interested in your goals and motivation. What's driving you right now in your athletic journey, and how clear do you feel about where you're heading?",
            f"Recovery and balance are so important. How well do you feel you're managing the balance between your sport, rest, and other parts of your life right now?"
        ]
        return random.sample(fallbacks, min(4, len(fallbacks)))
    
    def phase_2_voice_assessment(self, profile: AthleteProfile, 
                               voice_questions: List[Dict[str, str]], 
                               responses: Dict[str, str]) -> Dict[str, Any]:
        """Phase 2b: Conduct voice-based assessment conversation"""
        agent = self.create_human_coach_agent(profile, 'assessment')
        
        # Prepare response context
        response_context = self._build_response_context(voice_questions, responses)
        
        task = Task(
            description=f'''
            As Coach Alex, you've just finished asking {profile.name} personalized questions via voice,
            and they've provided thoughtful responses. Now create a natural, flowing conversation
            that acknowledges their responses and transitions to the coaching analysis phase.
            
            QUESTIONS ASKED AND RESPONSES RECEIVED:
            {response_context}
            
            YOUR VOICE CONVERSATION SHOULD:
            1. Thank them genuinely for their openness and honesty
            2. Acknowledge specific things they shared (reference 2-3 key points)
            3. Show you were truly listening: "What really stood out to me was..."
            4. Validate their feelings and experiences
            5. Express appreciation for their self-awareness
            6. Build their confidence: "I can already see some real strengths..."
            7. Create excitement for the analysis: "I'm putting together some ideas that I think will really help you"
            8. Transition naturally to analysis phase
            
            NATURAL SPEECH PATTERNS:
            - Use conversational connectors and natural pauses
            - Include empathetic responses to what they shared
            - Reference their specific sport and situation
            - Use their name 1-2 times naturally
            - Sound like you're processing their responses in real-time
            - Express genuine appreciation for their trust
            - Use encouraging language that builds anticipation
            
            CONVERSATION TONE:
            - Warm and appreciative
            - Professional but personal
            - Encouraging and supportive
            - Genuine interest in their success
            - Natural speech rhythm for voice delivery
            
            End with: "Give me just a moment to put together some personalized recommendations for you. 
            I'm really excited to share what I'm seeing and how we can help you take your mental game to the next level!"
            
            This should sound like a real coach who just learned a lot about their athlete 
            and is genuinely excited to help them improve.
            ''',
            expected_output='Natural voice conversation acknowledging responses and transitioning to coaching analysis',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        conversation_result = crew.kickoff()
        
        # Generate voice for the response acknowledgment
        voice_audio = self.generate_voice_audio(str(conversation_result), "assessment_acknowledgment")
        
        # Log assessment phase
        self.db_manager.log_consultation_phase(
            profile.session_id, 
            'voice_assessment_complete', 
            str(conversation_result)
        )
        
        return {
            'conversation': str(conversation_result),
            'audio_file': voice_audio,
            'responses_processed': len(responses)
        }
    
    def _build_response_context(self, questions: List[Dict[str, str]], responses: Dict[str, str]) -> str:
        """Build context from questions and responses"""
        context = ""
        for i, question_data in enumerate(questions):
            question_text = question_data['text']
            # Extract the actual question part
            if '?' in question_text:
                actual_question = question_text.split('?')[0] + '?'
                response = responses.get(f"question_{i+1}", "No response provided")
                context += f"\nQuestion {i+1}: {actual_question}\nResponse: {response}\n"
        
        return context
    
    def phase_3_voice_coaching_analysis(self, profile: AthleteProfile, 
                                      assessment_data: str) -> Dict[str, Any]:
        """Phase 3: Voice-delivered comprehensive coaching analysis"""
        agent = self.create_human_coach_agent(profile, 'coaching')
        
        # Compile comprehensive session data
        session_data = self._compile_session_data(profile, assessment_data)
        
        task = Task(
            description=f'''
            As Coach Alex, provide your complete coaching analysis and recommendations for {profile.name}.
            This will be delivered via voice, so speak naturally and conversationally.
            
            COMPLETE SESSION DATA:
            {session_data}
            
            VOICE COACHING ANALYSIS STRUCTURE:
            
            1. PERSONAL ACKNOWLEDGMENT (1 minute):
            "Alright {profile.name}, I've been thinking about everything you've shared with me, and 
            I have to say, I'm really impressed by your self-awareness and commitment..."
            - Reference specific things they mentioned
            - Acknowledge their honesty and openness
            - Express genuine appreciation for their trust
            
            2. KEY INSIGHTS (2 minutes):
            "Here's what I'm seeing as your biggest mental strengths..."
            - Highlight 2-3 specific mental strengths you identified
            - Connect their responses to positive patterns
            - Reference their sport and level specifically
            - Use encouraging, confidence-building language
            
            3. GROWTH OPPORTUNITIES (2 minutes):
            "Now, there are also some exciting areas where we can help you grow..."
            - Frame challenges as opportunities positively
            - Be specific about what you noticed
            - Connect to their goals and aspirations
            - Make it sound achievable and exciting
            
            4. PERSONALIZED MENTAL TRAINING PLAN (3 minutes):
            
            A) DAILY PRACTICES:
            "Let me give you some specific daily practices that I think will make a huge difference..."
            - Morning mental preparation (30 seconds)
            - Pre-training mindset routine (sport-specific)
            - Post-training reflection (quick and effective)
            - Evening mental recovery
            
            B) PERFORMANCE STRATEGIES:
            "For when you're competing or performing at your best..."
            - Competition mental preparation
            - Pressure management techniques
            - Confidence boosters (specific to their needs)
            - Focus enhancement methods
            
            C) LIFESTYLE INTEGRATION:
            "And here's how to weave this into your daily life..."
            - Stress management outside sport
            - Recovery optimization
            - Support system utilization
            
            5. MOTIVATION AND NEXT STEPS (1 minute):
            "Here's what I want you to remember, {profile.name}..."
            - Personal motivational message
            - Specific first steps for next 24 hours
            - Weekly practices to establish
            - Confidence booster for their journey
            
            6. CLOSING ENCOURAGEMENT (1 minute):
            "I'm really excited about your journey ahead..."
            - Express genuine belief in their potential
            - Offer ongoing support and encouragement
            - Make them feel empowered and motivated
            - End with inspiring, memorable message
            
            VOICE DELIVERY REQUIREMENTS:
            - Sound like a coach who genuinely cares about their success
            - Use natural speech rhythm and conversational tone
            - Include appropriate pauses and emphasis
            - Reference their specific situation throughout
            - Use encouraging, motivational language
            - Speak with confidence and expertise
            - Make it personal and memorable
            - Include specific, actionable advice
            - Sound genuinely excited about helping them
            
            HUMAN TOUCH:
            - Use personal anecdotes when appropriate ("I've worked with athletes who...")
            - Include relatable examples from their sport
            - Show genuine emotion and investment
            - Use natural speech patterns and phrases
            - Make them feel special and capable
            
            Remember: This is being spoken aloud, so make every word count and sound natural!
            ''',
            expected_output='Comprehensive voice coaching analysis with personalized recommendations',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        coaching_result = crew.kickoff()
        
        # Generate voice for coaching analysis
        voice_audio = self.generate_voice_audio(str(coaching_result), "coaching_analysis")
        
        # Save complete session
        self.db_manager.save_athlete_profile(profile)
        self.db_manager.log_consultation_phase(
            profile.session_id, 
            'voice_coaching_complete', 
            str(coaching_result)
        )
        
        return {
            'analysis': str(coaching_result),
            'audio_file': voice_audio,
            'session_complete': True
        }
    
    def _compile_session_data(self, profile: AthleteProfile, assessment_data: str) -> str:
        """Compile complete session data for analysis"""
        return f'''
        ATHLETE PROFILE SUMMARY:
        - Name: {profile.name}
        - Age: {profile.age} years ({self._get_age_category(profile.age)})
        - Physical: {profile.height}cm, {profile.weight}kg (BMI: {self._calculate_bmi(profile.weight, profile.height):.1f})
        - Sport: {profile.sport_type.value}
        - Experience: {profile.athlete_level.value}
        - Training: {profile.training_frequency} sessions/week ({self._assess_training_load(profile.training_frequency)})
        - Session: {profile.timestamp.strftime('%B %d, %Y at %H:%M')}
        - Goals: {', '.join(profile.primary_goals) if profile.primary_goals else 'General improvement'}
        
        CONTEXTUAL ANALYSIS:
        - Age factors: {self._get_age_considerations(profile.age)}
        - Sport mental aspects: {self._get_sport_mental_aspects(profile.sport_type)}
        - Experience considerations: {self._get_level_considerations(profile.athlete_level)}
        - Current season: {self._get_current_season()}
        
        ASSESSMENT RESPONSES:
        {chr(10).join([f"Response to Question {i+1}: {r}" for i, r in enumerate(profile.current_responses.values())])}
        
        CONVERSATION FLOW:
        {assessment_data}
        '''
    
    def generate_voice_audio(self, text: str, filename_prefix: str = "coach_audio") -> Optional[str]:
        """Generate voice audio file and return path"""
        try:
            # Use gTTS for better quality (requires internet)
            audio_data = self.voice_service.text_to_speech_gtts(text)
            if audio_data:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', prefix=f"{filename_prefix}_")
                temp_file.write(audio_data)
                temp_file.close()
                return temp_file.name
            else:
                # Fallback to pyttsx3 (offline)
                return self.voice_service.text_to_speech_pyttsx3(text)
                
        except Exception as e:
            st.error(f"Voice generation error: {e}")
            # Try offline backup
            try:
                return self.voice_service.text_to_speech_pyttsx3(text)
            except Exception as e2:
                st.error(f"Backup TTS also failed: {e2}")
                return None
    
    def conduct_full_voice_consultation(self, profile: AthleteProfile) -> Dict[str, Any]:
        """Conduct complete voice-interactive consultation"""
        self.active_sessions[profile.session_id] = profile
        
        results = {
            'session_id': profile.session_id,
            'athlete_name': profile.name,
            'phases': {}
        }
        
        try:
            # Phase 1: Voice welcome
            welcome_result = self.phase_1_voice_welcome(profile)
            results['phases']['welcome'] = welcome_result
            
            # Phase 2a: Generate voice questions
            voice_questions = self.phase_2_voice_questions_generation(profile)
            results['phases']['questions'] = voice_questions
            
            # Note: Phase 2b (collecting responses) and Phase 3 will be handled by UI
            results['status'] = 'ready_for_interaction'
            
        except Exception as e:
            st.error(f"Consultation error: {e}")
            results['error'] = str(e)
        
        return results
    
    # Helper methods
    def _get_age_category(self, age: int) -> str:
        if age <= 18: return "Youth athlete"
        elif age <= 25: return "Young adult athlete"
        elif age <= 40: return "Adult athlete"
        else: return "Masters athlete"
    
    def _calculate_bmi(self, weight: float, height: float) -> float:
        return weight / ((height/100) ** 2)
    
    def _assess_training_load(self, frequency: int) -> str:
        if frequency <= 2: return "Light training load"
        elif frequency <= 4: return "Moderate training load"
        elif frequency <= 6: return "High training load"
        else: return "Very high training load"
    
    def _get_current_season(self) -> str:
        month = datetime.now().month
        if month in [12, 1, 2]: return "Winter training phase"
        elif month in [3, 4, 5]: return "Spring preparation phase"
        elif month in [6, 7, 8]: return "Summer competition phase"
        else: return "Fall transition phase"
    
    def _get_age_considerations(self, age: int) -> str:
        if age <= 18:
            return "School balance, peer relationships, parental expectations, identity development"
        elif age <= 25:
            return "Career establishment, independence, relationship development, financial pressures"
        elif age <= 40:
            return "Work-life balance, family responsibilities, time optimization, career peak"
        else:
            return "Health maintenance, injury prevention, wisdom sharing, legacy concerns"
    
    def _get_sport_mental_aspects(self, sport_type: SportType) -> str:
        aspects = {
            SportType.ENDURANCE: "Mental toughness, pain tolerance, pacing strategy, long-term focus",
            SportType.STRENGTH: "Progressive confidence, body awareness, fear management, personal records",
            SportType.TEAM_SPORTS: "Communication, team dynamics, leadership, shared responsibility",
            SportType.INDIVIDUAL: "Self-reliance, pressure handling, perfectionism, independent motivation",
            SportType.COMBAT: "Controlled aggression, fear management, tactical thinking, confidence under pressure",
            SportType.MIXED: "Adaptability, goal prioritization, diverse skill confidence, scheduling balance"
        }
        return aspects.get(sport_type, "General athletic mental performance")
    
    def _get_level_considerations(self, level: AthleteLevel) -> str:
        considerations = {
            AthleteLevel.BEGINNER: "Learning curve navigation, expectation management, foundation building",
            AthleteLevel.INTERMEDIATE: "Plateau breakthrough, skill refinement, goal evolution",
            AthleteLevel.ADVANCED: "Performance optimization, competition strategy, consistency development",
            AthleteLevel.ELITE: "Peak performance maintenance, pressure management, career sustainability"
        }
        return considerations.get(level, "Athletic development considerations")

# Streamlit UI for Voice Interaction
def main():
    st.set_page_config(
        page_title="🎤 Voice Mental Performance Coach", 
        page_icon="🎤🧠", 
        layout="wide"
    )
    
    # Custom CSS for voice interface
    st.markdown("""
    <style>
    .voice-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    .audio-controls {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .coach-message {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎤 Voice Mental Performance Coach")
    st.markdown("*Talk with Coach Alex - Your AI Mental Performance Specialist*")
    
    # Initialize system
    if 'voice_coach' not in st.session_state:
        with st.spinner("Initializing voice coach system..."):
            st.session_state.voice_coach = VoiceEnabledCoach()
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = None
    
    if 'consultation_stage' not in st.session_state:
        st.session_state.consultation_stage = 'setup'
    
    coach = st.session_state.voice_coach
    
    # Sidebar with voice instructions
    with st.sidebar:
        st.header("🎤 Voice Consultation Guide")
        st.markdown("""
        **🗣️ How This Works:**
        1. Set up your athlete profile
        2. Listen to Coach Alex's welcome
        3. Answer questions by voice or text
        4. Receive personalized voice coaching
        
        **🎧 Audio Requirements:**
        - Headphones recommended
        - Quiet environment
        - Clear microphone
        """)
        
        st.header("🔧 System Status")
        if hasattr(st.session_state, 'voice_coach'):
            st.success("✅ Voice Coach Ready")
            st.success("✅ TTS System Active")
            st.success("✅ Speech Recognition Ready")
        else:
            st.warning("⚠️ Initializing...")
        
        if st.session_state.current_profile:
            st.header("👤 Current Session")
            profile = st.session_state.current_profile
            st.write(f"**Athlete:** {profile.name}")
            st.write(f"**Sport:** {profile.sport_type.value}")
            st.write(f"**Started:** {profile.timestamp.strftime('%H:%M')}")
    
    # Main consultation flow
    if st.session_state.consultation_stage == 'setup':
        show_voice_profile_setup()
    elif st.session_state.consultation_stage == 'welcome':
        show_voice_welcome_phase()
    elif st.session_state.consultation_stage == 'assessment':
        show_voice_assessment_phase()
    elif st.session_state.consultation_stage == 'coaching':
        show_voice_coaching_phase()

def show_voice_profile_setup():
    """Voice-optimized athlete profile setup"""
    st.markdown("""
    <div class="voice-container">
        <h2>🎤 Welcome to Voice Coaching with Coach Alex</h2>
        <p>Get ready for a completely interactive, voice-based mental performance consultation. 
        Coach Alex will speak with you naturally, ask personalized questions, and provide 
        voice-delivered coaching recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("voice_athlete_profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Info")
            name = st.text_input("First Name", placeholder="What should Coach Alex call you?")
            age = st.number_input("Age", min_value=13, max_value=80, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
            height = st.number_input("Height (cm)", min_value=120, max_value=230, value=170)
        
        with col2:
            st.subheader("🏃‍♂️ Athletic Profile")
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
        
        st.subheader("🎯 Goals for Today's Session")
        goals_text = st.text_area(
            "What would you like to work on with Coach Alex?",
            placeholder="e.g., improve confidence, manage pre-competition nerves, enhance focus during training...",
            height=100
        )
        
        # Voice preference
        st.subheader("🔊 Voice Interaction Preference")
        voice_preference = st.radio(
            "How would you like to interact?",
            ["Voice + Text (Recommended)", "Voice Only", "Text Only"]
        )
        
        submitted = st.form_submit_button(
            "🎤 Start Voice Consultation with Coach Alex", 
            type="primary", 
            use_container_width=True
        )
        
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
                primary_goals=goals,
                preferences={'voice_interaction': voice_preference}
            )
            
            st.session_state.current_profile = profile
            st.session_state.consultation_stage = 'welcome'
            st.rerun()
        elif submitted:
            st.error("Please enter your name to begin the voice consultation.")

def show_voice_welcome_phase():
    """Voice welcome phase with audio playback"""
    st.header("👋 Phase 1: Welcome from Coach Alex")
    
    profile = st.session_state.current_profile
    coach = st.session_state.voice_coach
    
    if 'welcome_complete' not in st.session_state:
        with st.spinner("🎤 Coach Alex is preparing your personalized welcome..."):
            welcome_result = coach.phase_1_voice_welcome(profile)
            st.session_state.welcome_result = welcome_result
            st.session_state.welcome_complete = True
    
    welcome_data = st.session_state.welcome_result
    
    # Display coach message
    st.markdown(f"""
    <div class="coach-message">
        <h4>🎤 Coach Alex speaks:</h4>
        <p>{welcome_data['text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Audio playback
    if welcome_data.get('audio_file'):
        st.subheader("🔊 Listen to Coach Alex")
        
        try:
            with open(welcome_data['audio_file'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
        except Exception as e:
            st.warning(f"Audio playback issue: {e}")
    
    # Voice response option
    st.markdown("---")
    st.subheader("🎤 Your Response")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Respond by Voice", type="primary", use_container_width=True):
            voice_response = get_voice_response("Tell Coach Alex how you're feeling and if you're ready to begin:")
            if voice_response:
                st.session_state.welcome_response = voice_response
                st.success(f"✅ Your response: {voice_response}")
    
    with col2:
        text_response = st.text_area(
            "Or type your response:",
            placeholder="Tell Coach Alex how you're feeling and if you're ready to begin...",
            key="welcome_text_response"
        )
        if text_response:
            st.session_state.welcome_response = text_response
    
    # Continue button
    if 'welcome_response' in st.session_state:
        st.success("✅ Response recorded!")
        if st.button("Continue to Assessment Questions", type="primary", use_container_width=True):
            st.session_state.consultation_stage = 'assessment'
            st.rerun()

def show_voice_assessment_phase():
    """Voice assessment phase with interactive Q&A"""
    st.header("🧠 Phase 2: Voice Assessment with Coach Alex")
    
    profile = st.session_state.current_profile
    coach = st.session_state.voice_coach
    
    # Generate voice questions
    if 'voice_questions' not in st.session_state:
        with st.spinner("🎤 Coach Alex is preparing personalized questions for you..."):
            voice_questions = coach.phase_2_voice_questions_generation(profile)
            st.session_state.voice_questions = voice_questions
            st.session_state.current_question = 0
            st.session_state.assessment_responses = {}
    
    questions = st.session_state.voice_questions
    current_q = st.session_state.current_question
    
    if current_q < len(questions):
        question_data = questions[current_q]
        
        st.markdown(f"""
        <div class="coach-message">
            <h4>🎤 Coach Alex - Question {current_q + 1} of {len(questions)}:</h4>
            <p>{question_data['text']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Play question audio
        if question_data.get('audio_file'):
            try:
                with open(question_data['audio_file'], 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3', autoplay=True)
            except Exception as e:
                st.warning(f"Audio playback issue: {e}")
        
        # Response collection
        st.subheader("🎤 Your Response")
        
        col1, col2 = st.columns(2)
        
        response_key = f"response_{current_q}"
        
        with col1:
            if st.button("🎤 Answer by Voice", type="primary", use_container_width=True):
                voice_response = get_voice_response(f"Answer Coach Alex's question {current_q + 1}:")
                if voice_response:
                    st.session_state.assessment_responses[f"question_{current_q + 1}"] = voice_response
                    st.success(f"✅ Voice response recorded!")
                    
        with col2:
            text_response = st.text_area(
                "Or type your answer:",
                placeholder="Share your thoughts and feelings...",
                key=f"text_response_{current_q}",
                height=120
            )
            
            if text_response:
                st.session_state.assessment_responses[f"question_{current_q + 1}"] = text_response
        
        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_q > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.current_question = current_q - 1
                    st.rerun()
        
        with col2:
            if f"question_{current_q + 1}" in st.session_state.assessment_responses:
                if st.button("➡️ Next Question", type="primary", use_container_width=True):
                    st.session_state.current_question = current_q + 1
                    st.rerun()
            else:
                st.warning("Please provide a response to continue")
        
        with col3:
            responses_count = len(st.session_state.assessment_responses)
            if responses_count >= 4:  # Minimum 4 responses
                if st.button("✅ Complete Assessment", type="secondary", use_container_width=True):
                    finalize_assessment()
        
        # Progress indicator
        progress = len(st.session_state.assessment_responses) / len(questions)
        st.progress(progress, text=f"Progress: {len(st.session_state.assessment_responses)}/{len(questions)} questions answered")
        
    else:
        # All questions completed
        st.success("🎉 Assessment Complete!")
        finalize_assessment()

def finalize_assessment():
    """Finalize assessment and move to coaching phase"""
    profile = st.session_state.current_profile
    coach = st.session_state.voice_coach
    
    # Update profile with responses
    profile.current_responses = st.session_state.assessment_responses
    
    # Generate assessment acknowledgment
    if 'assessment_acknowledgment' not in st.session_state:
        with st.spinner("🎤 Coach Alex is processing your responses..."):
            questions = st.session_state.voice_questions
            assessment_result = coach.phase_2_voice_assessment(profile, questions, profile.current_responses)
            st.session_state.assessment_acknowledgment = assessment_result
    
    st.session_state.consultation_stage = 'coaching'
    st.rerun()

def show_voice_coaching_phase():
    """Voice coaching analysis and recommendations"""
    st.header("🏆 Phase 3: Your Voice Coaching Analysis")
    
    profile = st.session_state.current_profile
    coach = st.session_state.voice_coach
    
    # Show assessment acknowledgment first
    if 'assessment_acknowledgment' in st.session_state:
        ack_data = st.session_state.assessment_acknowledgment
        
        st.markdown(f"""
        <div class="coach-message">
            <h4>🎤 Coach Alex responds to your answers:</h4>
            <p>{ack_data['conversation']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if ack_data.get('audio_file'):
            st.audio(open(ack_data['audio_file'], 'rb').read())
    
    # Generate full coaching analysis
    if 'coaching_analysis' not in st.session_state:
        with st.spinner("🎤 Coach Alex is creating your personalized coaching plan..."):
            assessment_data = st.session_state.assessment_acknowledgment['conversation']
            coaching_result = coach.phase_3_voice_coaching_analysis(profile, assessment_data)
            st.session_state.coaching_analysis = coaching_result
    
    coaching_data = st.session_state.coaching_analysis
    
    st.markdown(f"""
    <div class="coach-message">
        <h4>🎤 Coach Alex - Your Complete Coaching Analysis:</h4>
        <div style="max-height: 400px; overflow-y: auto; padding: 1rem; background: white; border-radius: 8px; color: black;">
            {coaching_data['analysis'].replace(chr(10), '<br>')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Voice coaching delivery
    if coaching_data.get('audio_file'):
        st.subheader("🎧 Listen to Your Complete Coaching Plan")
        st.markdown("*Put on headphones for the best experience*")
        
        try:
            with open(coaching_data['audio_file'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
        except Exception as e:
            st.warning(f"Audio playback issue: {e}")
    
    # Session completion actions
    st.markdown("---")
    st.subheader("🎯 What's Next?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📱 Get Daily Audio Reminders", use_container_width=True):
            show_daily_audio_reminders(profile)
    
    with col2:
        if st.button("💾 Save Session", use_container_width=True):
            save_voice_session(profile)
    
    with col3:
        if st.button("🔄 New Voice Session", use_container_width=True):
            start_new_voice_session()
    
    # Session summary
    with st.expander("📊 Voice Session Summary"):
        st.write(f"**Athlete:** {profile.name}")
        st.write(f"**Session ID:** {profile.session_id[:8]}...")
        st.write(f"**Date:** {profile.timestamp.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Voice Questions:** {len(st.session_state.get('voice_questions', []))}")
        st.write(f"**Responses Given:** {len(profile.current_responses)}")
        st.write(f"**Goals Addressed:** {', '.join(profile.primary_goals)}")

def get_voice_response(prompt: str) -> Optional[str]:
    """Get voice response from user"""
    st.info(f"🎤 {prompt}")
    
    coach = st.session_state.voice_coach
    
    # Voice input button
    if st.button("🔴 Start Recording", type="primary"):
        try:
            with st.spinner("🎤 Listening for your response..."):
                response = coach.voice_service.listen_for_speech(timeout=15)
                if response:
                    st.success(f"✅ Heard: {response}")
                    return response
                else:
                    st.warning("No response detected. Please try again or use text input.")
                    return None
        except Exception as e:
            st.error(f"Voice input error: {e}")
            return None
    
    return None

def show_daily_audio_reminders(profile: AthleteProfile):
    """Generate daily audio reminders"""
    coach = st.session_state.voice_coach
    
    reminder_text = f"""
    Hi {profile.name}! This is Coach Alex with your daily mental training reminder.
    
    Take just 5 minutes today for your mental game:
    
    Morning: Start with three deep breaths and visualize one successful moment in your {profile.sport_type.value.lower()}.
    
    Pre-training: Remind yourself of one thing you're improving today.
    
    Post-training: Ask yourself what you learned about your mental game today.
    
    You've got this! Keep building that mental strength every single day.
    """
    
    st.markdown("""
    <div class="coach-message">
        <h4>📱 Daily Audio Reminder Created!</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate reminder audio
    with st.spinner("🎤 Creating your daily reminder audio..."):
        reminder_audio = coach.generate_voice_audio(reminder_text, "daily_reminder")
        
        if reminder_audio:
            st.success("✅ Your personal daily reminder from Coach Alex:")
            try:
                with open(reminder_audio, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
                
                st.download_button(
                    label="💾 Download Daily Reminder Audio",
                    data=audio_bytes,
                    file_name=f"daily_reminder_{profile.name}.mp3",
                    mime="audio/mp3"
                )
            except Exception as e:
                st.error(f"Audio download error: {e}")

def save_voice_session(profile: AthleteProfile):
    """Save voice session data"""
    coach = st.session_state.voice_coach
    
    try:
        coach.db_manager.save_athlete_profile(profile)
        st.success(f"✅ Session saved for {profile.name}!")
        
        # Show save summary
        st.markdown(f"""
        <div class="coach-message">
            <h4>💾 Session Saved Successfully</h4>
            <p><strong>Session ID:</strong> {profile.session_id[:8]}...</p>
            <p><strong>Date:</strong> {profile.timestamp.strftime('%B %d, %Y at %H:%M')}</p>
            <p><strong>Responses Saved:</strong> {len(profile.current_responses)}</p>
            <p><strong>Goals Addressed:</strong> {', '.join(profile.primary_goals)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Save error: {e}")

def start_new_voice_session():
    """Start a new voice consultation session"""
    # Clear session state except for the coach system
    for key in list(st.session_state.keys()):
        if key not in ['voice_coach']:
            del st.session_state[key]
    
    st.session_state.consultation_stage = 'setup'
    st.success("🔄 Ready for new voice consultation!")
    st.rerun()

def play_audio_file(file_path: str):
    """Play audio file using pygame"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"Audio playback error: {e}")

def create_voice_summary_audio(profile: AthleteProfile) -> Optional[str]:
    """Create a summary audio message"""
    coach = st.session_state.voice_coach
    
    summary_text = f"""
    {profile.name}, this is Coach Alex with a quick summary of our session today.
    
    We covered {len(profile.current_responses)} important areas of your mental game.
    You showed great self-awareness and openness during our conversation.
    
    Your main focus areas are: {', '.join(profile.primary_goals[:3]) if profile.primary_goals else 'building overall mental strength'}.
    
    Remember to practice the daily mental training techniques we discussed.
    Start with just 5 minutes a day, and build from there.
    
    I believe in your potential, and I'm excited to see how you grow.
    Keep up the great work, and remember - your mental game is just as important as your physical training.
    
    Take care, {profile.name}!
    """
    
    return coach.generate_voice_audio(summary_text, "session_summary")

# Enhanced Streamlit components
def display_voice_controls():
    """Display voice interaction controls"""
    st.markdown("""
    <div class="audio-controls">
        <h4>🎤 Voice Controls</h4>
        <p>• Click "Start Recording" and speak clearly</p>
        <p>• Use headphones to avoid audio feedback</p>
        <p>• Speak in a quiet environment for best results</p>
        <p>• You can always use text input as backup</p>
    </div>
    """, unsafe_allow_html=True)

def show_microphone_test():
    """Test microphone functionality"""
    st.subheader("🎤 Microphone Test")
    
    if st.button("Test Your Microphone"):
        coach = st.session_state.voice_coach
        
        with st.spinner("Testing microphone... say something!"):
            test_response = coach.voice_service.listen_for_speech(timeout=5)
            
            if test_response:
                st.success(f"✅ Microphone working! Heard: '{test_response}'")
            else:
                st.warning("⚠️ Microphone test failed. Please check your microphone settings.")

def show_voice_session_history():
    """Show previous voice sessions for returning athletes"""
    if 'current_profile' in st.session_state and st.session_state.current_profile:
        profile = st.session_state.current_profile
        coach = st.session_state.voice_coach
        
        history = coach.db_manager.get_athlete_history(profile.name)
        
        if len(history) > 1:
            st.subheader(f"📚 Previous Sessions for {profile.name}")
            
            for session in history[1:4]:  # Show last 3 previous sessions
                date = datetime.fromisoformat(session['date']).strftime('%B %d, %Y')
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>📅 {date}</strong><br>
                    Sport: {session['sport']}<br>
                    Goals: {', '.join(session['goals'][:2])}
                </div>
                """, unsafe_allow_html=True)

# Main application runner
if __name__ == "__main__":
    # Check system requirements
    st.markdown("---")
    with st.expander("🔧 Voice System Requirements", expanded=False):
        st.markdown("""
        **Required for Voice Functionality:**
        
        1. **Ollama with Llama 3.2:**
        ```bash
        # Install Ollama
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull the model
        ollama pull llama3.2:latest
        
        # Start Ollama server
        ollama serve
        ```
        
        2. **Python Voice Libraries:**
        ```bash
        pip install streamlit crewai pyttsx3 SpeechRecognition gTTS pygame pyaudio
        
        # For microphone support (Linux/Mac):
        sudo apt-get install portaudio19-dev python3-pyaudio  # Ubuntu/Debian
        brew install portaudio  # macOS
        ```
        
        3. **Audio Hardware:**
        - Working microphone for voice input
        - Speakers or headphones for audio output
        - Quiet environment for best speech recognition
        
        4. **Internet Connection:**
        - Required for Google TTS (gTTS)
        - Offline backup available with pyttsx3
        
        **🎤 Voice Features:**
        - Natural conversation flow
        - Human-like speech patterns
        - Interactive voice responses
        - Personalized audio coaching
        - Daily reminder generation
        """)
    
    # System status check
    st.markdown("### 🔍 System Status Check")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check Ollama connection
        try:
            import requests
            response = requests.get(f"{Config.API_BASE}/api/tags", timeout=3)
            if response.status_code == 200:
                st.success("✅ Ollama Connected")
            else:
                st.error("❌ Ollama Connection Failed")
        except:
            st.error("❌ Ollama Not Available")
    
    with col2:
        # Check TTS libraries
        try:
            import pyttsx3
            import gtts
            st.success("✅ TTS Libraries Ready")
        except ImportError as e:
            st.error(f"❌ Missing TTS: {e}")
    
    with col3:
        # Check Speech Recognition
        try:
            import speech_recognition
            st.success("✅ Speech Recognition Ready")
        except ImportError as e:
            st.error(f"❌ Missing Speech Recognition: {e}")
    
    # Run main application
    main()

# Additional utility functions for voice interaction
def create_voice_interaction_session():
    """Create an interactive voice session"""
    st.markdown("""
    ### 🎤 Voice Interaction Session
    
    **How to have the best voice conversation with Coach Alex:**
    
    1. **Environment Setup:**
       - Find a quiet space
       - Use headphones if possible
       - Test your microphone first
    
    2. **Speaking Tips:**
       - Speak clearly and at normal pace
       - Take your time to think before responding
       - It's okay to pause - Coach Alex will wait
    
    3. **Technical Notes:**
       - Voice recognition works best with clear audio
       - Text backup is always available
       - Audio files can be downloaded for later
    """)

def generate_voice_coaching_summary():
    """Generate a voice summary of the entire session"""
    if 'current_profile' in st.session_state and 'coaching_analysis' in st.session_state:
        profile = st.session_state.current_profile
        
        with st.spinner("🎤 Creating your session summary audio..."):
            summary_audio = create_voice_summary_audio(profile)
            
            if summary_audio:
                st.subheader("🎧 Session Summary from Coach Alex")
                try:
                    with open(summary_audio, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                        
                        st.download_button(
                            label="💾 Download Session Summary",
                            data=audio_bytes,
                            file_name=f"coaching_summary_{profile.name}_{datetime.now().strftime('%Y%m%d')}.mp3",
                            mime="audio/mp3"
                        )
                except Exception as e:
                    st.error(f"Summary audio error: {e}")

# Voice-specific helper functions
def check_audio_permissions():
    """Check if audio permissions are available"""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
        return True
    except Exception as e:
        st.error(f"Audio permission error: {e}")
        return False

def create_voice_feedback_loop():
    """Create interactive voice feedback system"""
    if st.button("🎤 Give Voice Feedback to Coach Alex"):
        feedback = get_voice_response("Share your feedback about this voice coaching session:")
        if feedback:
            st.success("✅ Thank you for your feedback!")
            st.write(f"**Your feedback:** {feedback}")
            
            # Save feedback to database
            if 'current_profile' in st.session_state:
                profile = st.session_state.current_profile
                coach = st.session_state.voice_coach
                coach.db_manager.log_consultation_phase(
                    profile.session_id, 
                    'voice_feedback', 
                    feedback
                )

# Error handling and fallbacks
def handle_voice_errors():
    """Handle common voice interaction errors"""
    st.markdown("""
    ### 🔧 Voice Troubleshooting
    
    **If voice isn't working:**
    1. Check microphone permissions in your browser
    2. Try refreshing the page
    3. Use text input as backup
    4. Ensure quiet environment
    
    **Audio playback issues:**
    1. Check speaker/headphone connection
    2. Verify browser audio settings
    3. Try downloading audio files directly
    """)

# Run the enhanced voice application
if __name__ == "__main__":
    main()