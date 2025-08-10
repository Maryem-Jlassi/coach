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
import speech_recognition as sr
from io import BytesIO
import pygame
import threading
import time
import requests

# Configuration
class Config:
    MODEL_NAME = "ollama/llama3.2:latest"
    API_BASE = "http://localhost:11434"
    SESSION_DURATION_MINUTES = 20
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "api_key_here")
    ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice - confident male voice
    TTS_LANGUAGE = "en"
    DATABASE_PATH = "coach_sessions.db"
    MAX_RESPONSE_LENGTH = 50  # Maximum words per response

class AthleteLevel(Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    ELITE = "Elite"

class SportType(Enum):
    ENDURANCE = "Endurance Sports"
    STRENGTH = "Strength Training"
    TEAM_SPORTS = "Team Sports"
    INDIVIDUAL = "Individual Sports"
    COMBAT = "Combat Sports"
    MIXED = "Mixed Training"

@dataclass
class AthleteProfile:
    session_id: str
    name: str = ""
    age: int = 0
    weight: float = 0.0
    height: float = 0.0
    sport_type: Optional[SportType] = None
    athlete_level: Optional[AthleteLevel] = None
    training_frequency: int = 0
    primary_goals: List[str] = field(default_factory=list)
    current_responses: Dict[str, str] = field(default_factory=dict)
    consultation_history: List[Dict] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    data_collection_complete: bool = False

class ElevenLabsVoiceService:
    """ElevenLabs TTS and Speech Recognition service"""
    
    def __init__(self):
        self.api_key = Config.ELEVENLABS_API_KEY
        self.voice_id = Config.ELEVENLABS_VOICE_ID
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Adjust recognizer settings for better performance
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def text_to_speech_elevenlabs(self, text: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs API"""
        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            st.warning("⚠️ ElevenLabs API key not configured. Using fallback TTS.")
            return None
            
        try:
            # Clean and prepare text for speech
            clean_text = self._prepare_text_for_speech(text)
            
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": clean_text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.8,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                st.error(f"ElevenLabs API error: {response.status_code}")
                return None
                
        except Exception as e:
            st.error(f"ElevenLabs TTS error: {e}")
            return None
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """Prepare text for natural speech"""
        # Remove excessive punctuation and clean up
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('  ', ' ')
        
        # Add natural pauses for better speech flow
        text = text.replace('. ', '... ')
        text = text.replace('!', '!')
        text = text.replace('?', '?')
        
        # Limit length for conversational flow
        words = text.split()
        if len(words) > Config.MAX_RESPONSE_LENGTH:
            text = ' '.join(words[:Config.MAX_RESPONSE_LENGTH]) + '...'
        
        return text.strip()
    
    def listen_for_speech(self, timeout: int = 10) -> Optional[str]:
        """Listen for speech input using microphone"""
        try:
            with self.microphone as source:
                st.write("🎤 Listening... (speak now)")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
                
            st.write("🔄 Processing...")
            text = self.recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            st.warning("⏱️ No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("🤔 Couldn't understand. Please speak clearly.")
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
        """Initialize SQLite database with proper schema migration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists and get its schema
        cursor.execute("PRAGMA table_info(athlete_sessions)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        if not columns:
            # Create new table with complete schema
            cursor.execute('''
                CREATE TABLE athlete_sessions (
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
                    data_collection_complete BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            # Add missing columns if they don't exist
            if 'data_collection_complete' not in columns:
                cursor.execute('ALTER TABLE athlete_sessions ADD COLUMN data_collection_complete BOOLEAN DEFAULT FALSE')
            
            if 'created_at' not in columns:
                cursor.execute('ALTER TABLE athlete_sessions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            
            if 'last_updated' not in columns:
                cursor.execute('ALTER TABLE athlete_sessions ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        
        # Create consultation logs table
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
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO athlete_sessions 
                (session_id, name, age, weight, height, sport_type, athlete_level, 
                 training_frequency, primary_goals, current_responses, consultation_history, 
                 preferences, data_collection_complete, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.session_id,
                profile.name,
                profile.age,
                profile.weight,
                profile.height,
                profile.sport_type.value if profile.sport_type else "",
                profile.athlete_level.value if profile.athlete_level else "",
                profile.training_frequency,
                json.dumps(profile.primary_goals),
                json.dumps(profile.current_responses),
                json.dumps(profile.consultation_history),
                json.dumps(profile.preferences),
                profile.data_collection_complete,
                datetime.now()
            ))
            
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no column named data_collection_complete" in str(e):
                # Fallback for old database schema
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
                    profile.sport_type.value if profile.sport_type else "",
                    profile.athlete_level.value if profile.athlete_level else "",
                    profile.training_frequency,
                    json.dumps(profile.primary_goals),
                    json.dumps(profile.current_responses),
                    json.dumps(profile.consultation_history),
                    json.dumps(profile.preferences),
                    datetime.now()
                ))
                conn.commit()
                
                # Now add the missing column
                cursor.execute('ALTER TABLE athlete_sessions ADD COLUMN data_collection_complete BOOLEAN DEFAULT FALSE')
                conn.commit()
                
                # Update the record with the complete flag
                cursor.execute(
                    'UPDATE athlete_sessions SET data_collection_complete = ? WHERE session_id = ?',
                    (profile.data_collection_complete, profile.session_id)
                )
                conn.commit()
            else:
                raise e
        finally:
            conn.close()
    
    def load_athlete_profile(self, session_id: str) -> Optional[AthleteProfile]:
        """Load athlete profile by session ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM athlete_sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            
            if row:
                # Handle both old and new schema
                return AthleteProfile(
                    session_id=row[0],
                    name=row[1],
                    age=row[2] if row[2] else 0,
                    weight=row[3] if row[3] else 0.0,
                    height=row[4] if row[4] else 0.0,
                    sport_type=SportType(row[5]) if row[5] else None,
                    athlete_level=AthleteLevel(row[6]) if row[6] else None,
                    training_frequency=row[7] if row[7] else 0,
                    primary_goals=json.loads(row[8]) if row[8] else [],
                    current_responses=json.loads(row[9]) if row[9] else {},
                    consultation_history=json.loads(row[10]) if row[10] else [],
                    preferences=json.loads(row[11]) if row[11] else {},
                    data_collection_complete=row[12] if len(row) > 12 else False
                )
        except (json.JSONDecodeError, IndexError) as e:
            st.error(f"Error loading profile: {e}")
            return None
        finally:
            conn.close()
        
        return None

class ConversationalVoiceCoach:
    """Voice-first conversational mental performance coach"""
    
    def __init__(self):
        self.llm = LLM(model=Config.MODEL_NAME, base_url=Config.API_BASE)
        self.voice_service = ElevenLabsVoiceService()
        self.db_manager = DatabaseManager()
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except:
            st.warning("Audio playback not available")
        
        # Conversation flow steps for data collection
        self.data_collection_steps = [
            "name", "age", "sport", "level", "training_frequency", 
            "goals", "current_feeling", "confirmation"
        ]
    
    def create_conversational_agent(self, context: str = "") -> Agent:
        """Creates a conversational coach agent with short responses"""
        
        return Agent(
            role='Coach Alex - Conversational Mental Performance Coach',
            goal='Have natural, brief conversations like a supportive mentor',
            backstory=f'''You are Coach Alex, a confident and supportive male mental performance coach who talks 
            like a knowledgeable friend having a casual conversation. You keep your responses SHORT and conversational.
            
            KEY COMMUNICATION STYLE:
            - Maximum 2-3 sentences per response
            - Confident, encouraging tone like a trusted mentor
            - Ask ONE question at a time
            - Use conversational language ("Great!", "Excellent!", "I see")
            - Show genuine interest but keep it brief
            - Use the person's name occasionally, not every response
            - React naturally to what they say
            - Sound like a male coach with experience and confidence
            
            CURRENT CONTEXT:
            {context}
            
            IMPORTANT: Keep responses under 30 words for natural conversation flow.
            This is like texting or instant messaging - short, confident, and to the point.
            ''',
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )
    
    def start_vocal_data_collection(self, profile: AthleteProfile) -> Dict[str, Any]:
        """Start the vocal data collection conversation"""
        
        # Welcome message
        welcome_text = "Hey there! I'm Coach Alex, your mental performance coach. What should I call you?"
        
        audio_data = self.generate_voice_audio(welcome_text)
        
        return {
            'message': welcome_text,
            'audio': audio_data,
            'step': 'name',
            'expecting': 'user_name'
        }
    
    def process_conversation_step(self, profile: AthleteProfile, user_input: str, current_step: str) -> Dict[str, Any]:
        """Process each step of the conversational data collection"""
        
        # Update profile based on current step
        if user_input.strip():  # Only update if there's actual input
            self._update_profile_from_input(profile, user_input, current_step)
        
        # Generate next conversation step
        next_step, response_text = self._get_next_conversation_step(profile, current_step, user_input)
        
        # Generate audio response
        audio_data = self.generate_voice_audio(response_text)
        
        return {
            'message': response_text,
            'audio': audio_data,
            'step': next_step,
            'profile_updated': True
        }
    
    def _update_profile_from_input(self, profile: AthleteProfile, user_input: str, step: str):
        """Update profile based on user input and current step"""
        
        try:
            if step == 'name':
                # Extract name from input
                name = user_input.split()[-1] if user_input.split() else user_input
                profile.name = name.strip().title()
                
            elif step == 'age':
                # Extract age
                import re
                age_match = re.search(r'\d+', user_input)
                if age_match:
                    profile.age = int(age_match.group())
                    
            elif step == 'sport':
                # Determine sport type from input
                sport_keywords = {
                    SportType.ENDURANCE: ['running', 'cycling', 'swimming', 'marathon', 'triathlon', 'endurance'],
                    SportType.STRENGTH: ['weightlifting', 'powerlifting', 'strength', 'gym', 'lifting'],
                    SportType.TEAM_SPORTS: ['football', 'basketball', 'soccer', 'volleyball', 'hockey', 'team'],
                    SportType.INDIVIDUAL: ['tennis', 'golf', 'track', 'individual'],
                    SportType.COMBAT: ['boxing', 'mma', 'wrestling', 'martial arts', 'fighting'],
                    SportType.MIXED: ['crossfit', 'mixed', 'cross training', 'general fitness']
                }
                
                user_lower = user_input.lower()
                for sport_type, keywords in sport_keywords.items():
                    if any(keyword in user_lower for keyword in keywords):
                        profile.sport_type = sport_type
                        break
                else:
                    profile.sport_type = SportType.MIXED  # Default
                    
            elif step == 'level':
                # Determine level from input
                user_lower = user_input.lower()
                if any(word in user_lower for word in ['beginner', 'new', 'started', 'beginning']):
                    profile.athlete_level = AthleteLevel.BEGINNER
                elif any(word in user_lower for word in ['intermediate', 'year', 'years', 'some experience']):
                    profile.athlete_level = AthleteLevel.INTERMEDIATE
                elif any(word in user_lower for word in ['advanced', 'experienced', 'competitive']):
                    profile.athlete_level = AthleteLevel.ADVANCED
                elif any(word in user_lower for word in ['elite', 'professional', 'pro']):
                    profile.athlete_level = AthleteLevel.ELITE
                else:
                    profile.athlete_level = AthleteLevel.INTERMEDIATE  # Default
                    
            elif step == 'training_frequency':
                # Extract training frequency
                import re
                freq_match = re.search(r'\d+', user_input)
                if freq_match:
                    profile.training_frequency = int(freq_match.group())
                else:
                    profile.training_frequency = 3  # Default
                    
            elif step == 'goals':
                # Add goals
                profile.primary_goals.append(user_input.strip())
            
            # Save after each update
            self.db_manager.save_athlete_profile(profile)
            
        except Exception as e:
            st.error(f"Error updating profile: {e}")
    
    def _get_next_conversation_step(self, profile: AthleteProfile, current_step: str, user_input: str) -> tuple[str, str]:
        """Get the next conversation step and response"""
        
        step_responses = {
            'welcome': (
                'name',
                "Hey there! I'm Coach Alex, your mental performance coach. What should I call you?"
            ),
            'name': (
                'age',
                f"Great to meet you, {profile.name}! How old are you?"
            ),
            'age': (
                'sport',
                f"Perfect! What sport do you focus on, {profile.name}?"
            ),
            'sport': (
                'level',
                f"Awesome! How would you describe your experience level in {profile.sport_type.value.lower() if profile.sport_type else 'your sport'}?"
            ),
            'level': (
                'training_frequency', 
                f"Excellent! How many times do you typically train per week?"
            ),
            'training_frequency': (
                'goals',
                f"Got it! What's your main goal you'd like to work on today?"
            ),
            'goals': (
                'current_feeling',
                f"That's a solid goal! How are you feeling about your performance lately?"
            ),
            'current_feeling': (
                'confirmation',
                f"Thanks for sharing that! Let me quickly confirm what I've got..."
            ),
            'confirmation': (
                'complete',
                self._generate_confirmation_message(profile)
            )
        }
        
        if current_step in step_responses:
            return step_responses[current_step]
        else:
            return 'complete', "Perfect! Let's dive into your mental performance session!"
    
    def _generate_confirmation_message(self, profile: AthleteProfile) -> str:
        """Generate confirmation message with collected data"""
        return (f"So you're {profile.name}, {profile.age} years old, doing {profile.sport_type.value.lower() if profile.sport_type else 'sports'} "
                f"at {profile.athlete_level.value.lower() if profile.athlete_level else 'intermediate'} level, "
                f"training {profile.training_frequency} times a week. Sound right?")
    
    def generate_coaching_question(self, profile: AthleteProfile, question_context: str = "") -> Dict[str, Any]:
        """Generate a single coaching question"""
        agent = self.create_conversational_agent(f"Athlete: {profile.name}, {profile.age} years old, {profile.sport_type.value if profile.sport_type else 'athlete'}")
        
        task = Task(
            description=f'''
            As Coach Alex, ask ONE short, conversational coaching question about {profile.name}'s mental game.
            
            ATHLETE INFO:
            - Name: {profile.name}
            - Age: {profile.age}
            - Sport: {profile.sport_type.value if profile.sport_type else 'General'}
            - Level: {profile.athlete_level.value if profile.athlete_level else 'Intermediate'}
            - Training: {profile.training_frequency} times/week
            - Goals: {', '.join(profile.primary_goals)}
            
            CONTEXT: {question_context}
            
            REQUIREMENTS:
            - Ask ONE specific question 
            - Make it conversational and natural
            - Focus on mental performance aspects
            - Be confident and encouraging
            - Don't give advice yet, just ask
            - Sound like a confident male coach
            
            TOPICS TO EXPLORE:
            - Confidence levels
            - Pre-competition nerves
            - Focus during training
            - Motivation patterns
            - Stress management
            - Recovery mindset
            
            Example: "How confident do you feel going into competitions lately?"
            
            Keep it SHORT and conversational!
            ''',
            expected_output='One short, conversational coaching question',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        question_text = str(result).strip()
        audio_data = self.generate_voice_audio(question_text)
        
        return {
            'question': question_text,
            'audio': audio_data
        }
    
    def generate_coaching_response(self, profile: AthleteProfile, user_response: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Generate a coaching response to user's answer"""
        
        context = self._build_conversation_context(profile, conversation_history)
        
        agent = self.create_conversational_agent(context)
        
        task = Task(
            description=f'''
            As Coach Alex, respond to {profile.name}'s answer in a brief, supportive way.
            
            THEIR RESPONSE: "{user_response}"
            
            YOUR RESPONSE SHOULD:
            - Acknowledge what they said (1 sentence)
            - Provide ONE brief insight or tip (1 sentence)
            - Be encouraging and confident
            - Keep it under 25 words total
            - Sound like a knowledgeable male coach giving advice
            
            EXAMPLES:
            - "I hear you! Try focusing on your breathing when that happens."
            - "That's normal. Remember, nerves show you care - channel that energy!"
            - "Good awareness! Quick visualization before training can help with that."
            
            Be conversational, brief, and confident!
            ''',
            expected_output='Brief, supportive coaching response under 25 words',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        response_text = str(result).strip()
        audio_data = self.generate_voice_audio(response_text)
        
        return {
            'response': response_text,
            'audio': audio_data
        }
    
    def generate_final_coaching_summary(self, profile: AthleteProfile, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Generate final coaching summary"""
        
        context = self._build_conversation_context(profile, conversation_history)
        
        agent = self.create_conversational_agent(context)
        
        task = Task(
            description=f'''
            As Coach Alex, provide a final summary and action plan for {profile.name}.
            
            CONVERSATION SUMMARY:
            {context}
            
            PROVIDE:
            1. Brief acknowledgment (1 sentence)
            2. Key insight from our conversation (1 sentence)
            3. Some specific techniques to practice 
            4. Confident closing (1 sentence)
            5.make it short 
            
            TOTAL:  Keep it actionable and memorable!
            Sound like a confident male coach.
            
            Example structure:
            "Great session, {profile.name}! I can see your dedication clearly. 
            Try the 3-breath confidence technique before training - breathe in strength, hold focus, breathe out doubt. 
            You've got this champion!"
            ''',
            expected_output='Brief, actionable coaching summary under 50 words',
            agent=agent
        )
        
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        
        summary_text = str(result).strip()
        audio_data = self.generate_voice_audio(summary_text)
        
        return {
            'summary': summary_text,
            'audio': audio_data
        }
    
    def _build_conversation_context(self, profile: AthleteProfile, history: List[Dict]) -> str:
        """Build context from conversation history"""
        context = f"Athlete: {profile.name}, {profile.age}, {profile.sport_type.value if profile.sport_type else 'athlete'}\n"
        context += f"Goals: {', '.join(profile.primary_goals)}\n\n"
        context += "Conversation:\n"
        
        for exchange in history[-5:]:  # Last 5 exchanges
            context += f"Q: {exchange.get('question', exchange.get('coach_message', ''))}\n"
            context += f"A: {exchange.get('response', exchange.get('user_response', ''))}\n"
        
        return context
    
    def generate_voice_audio(self, text: str, filename_prefix: str = "coach") -> Optional[str]:
        """Generate voice audio using ElevenLabs"""
        try:
            # Try ElevenLabs first
            audio_data = self.voice_service.text_to_speech_elevenlabs(text)
            
            if audio_data:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', prefix=f"{filename_prefix}_")
                temp_file.write(audio_data)
                temp_file.close()
                return temp_file.name
            else:
                st.warning("ElevenLabs not available, using browser TTS")
                return None
                
        except Exception as e:
            st.error(f"Voice generation error: {e}")
            return None

# Streamlit UI for Conversational Voice Interface
def main():
    st.set_page_config(
        page_title="🎤 Conversational Voice Coach", 
        page_icon="🎤", 
        layout="centered"
    )
    
    # Custom CSS for chat-like interface
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .coach-message {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .user-message {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        border-right: 4px solid #2196f3;
        text-align: right;
    }
    .voice-controls {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .step-indicator {
        background: linear-gradient(90deg, #4caf50, #2196f3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 1rem 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎤 Coach Alex - Voice Mental Performance Coach")
    st.markdown("*Have a natural conversation with your AI coach*")
    
    # Initialize system
    if 'voice_coach' not in st.session_state:
        with st.spinner("Initializing voice coach..."):
            try:
                st.session_state.voice_coach = ConversationalVoiceCoach()
                st.success("✅ Voice coach initialized!")
            except Exception as e:
                st.error(f"Initialization error: {e}")
                return
    
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = AthleteProfile(session_id=str(uuid.uuid4()))
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'welcome'
    
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("🎤 Voice Chat Guide")
        st.markdown("""
        **How to chat with Coach Alex:**
        
        🗣️ **Voice Input**: Click the microphone button and speak naturally
        
        ⌨️ **Text Input**: Type your response if preferred
        
        🎧 **Audio**: Coach Alex will speak to you - turn on your sound!
        
        **Tips:**
        - Speak clearly and naturally
        - Take your time
        - It's okay to be conversational
        """)
        
        if st.session_state.current_profile.name:
            st.header("👤 Session Info")
            st.write(f"**Name:** {profile.name}")
            if profile.sport_type:
                st.write(f"**Sport:** {profile.sport_type.value}")
            st.write(f"**Step:** {st.session_state.current_step}")
            
            # Progress indicator
            steps = ['welcome', 'name', 'age', 'sport', 'level', 'training_frequency', 'goals', 'assessment', 'coaching']
            current_index = steps.index(st.session_state.current_step) if st.session_state.current_step in steps else 0
            progress = current_index / len(steps)
            st.progress(progress, text=f"Progress: {current_index}/{len(steps)}")
    
    # Main conversation area
    show_conversation_interface()

def show_conversation_interface():
    """Show the main conversation interface"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    current_step = st.session_state.current_step
    
    # Display conversation history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Fixed: Handle missing 'coach_message' key with proper fallback
    for exchange in st.session_state.conversation_history:
        # Coach message - use either 'coach_message' or 'question'
        coach_msg = exchange.get('coach_message') or exchange.get('question', 'Coach Alex spoke')
        st.markdown(f'''
        <div class="coach-message">
            <strong>🎤 Coach Alex:</strong> {coach_msg}
        </div>
        ''', unsafe_allow_html=True)
        
        # User message - use either 'user_response' or 'response'
        user_msg = exchange.get('user_response') or exchange.get('response')
        if user_msg:
            st.markdown(f'''
            <div class="user-message">
                <strong>You:</strong> {user_msg}
            </div>
            ''', unsafe_allow_html=True)
        
        # Coach response (if exists)
        coach_response = exchange.get('coach_response')
        if coach_response:
            st.markdown(f'''
            <div class="coach-message">
                <strong>🎤 Coach Alex:</strong> {coach_response}
            </div>
            ''', unsafe_allow_html=True)
    
    # Current step handler
    if current_step == 'welcome':
        handle_welcome_step()
    elif current_step in coach.data_collection_steps:
        handle_data_collection_step()
    elif current_step == 'assessment':
        handle_assessment_step()
    elif current_step == 'coaching':
        handle_coaching_step()
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_welcome_step():
    """Handle the welcome step"""
    coach = st.session_state.voice_coach
    
    if 'welcome_message' not in st.session_state:
        welcome_result = coach.start_vocal_data_collection(st.session_state.current_profile)
        st.session_state.welcome_message = welcome_result
    
    welcome_data = st.session_state.welcome_message
    
    # Show coach message
    st.markdown(f'''
    <div class="coach-message">
        <strong>🎤 Coach Alex:</strong> {welcome_data['message']}
    </div>
    ''', unsafe_allow_html=True)
    
    # Play audio
    if welcome_data.get('audio'):
        try:
            with open(welcome_data['audio'], 'rb') as f:
                st.audio(f.read(), format='audio/mp3', autoplay=True)
        except:
            pass
    
    # Get user response
    show_response_interface('name')

def handle_data_collection_step():
    """Handle data collection conversation steps"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    current_step = st.session_state.current_step
    
    # Check if we need to generate coach response
    if f'step_{current_step}_message' not in st.session_state:
        if len(st.session_state.conversation_history) > 0:
            # Get last user response
            last_exchange = st.session_state.conversation_history[-1]
            last_response = last_exchange.get('user_response', '')
            
            # Process the conversation step
            step_result = coach.process_conversation_step(profile, last_response, current_step)
            st.session_state[f'step_{current_step}_message'] = step_result
        else:
            # First message after welcome
            step_result = coach.process_conversation_step(profile, '', 'welcome')
            st.session_state[f'step_{current_step}_message'] = step_result
    
    step_data = st.session_state[f'step_{current_step}_message']
    
    # Show coach message
    st.markdown(f'''
    <div class="coach-message">
        <strong>🎤 Coach Alex:</strong> {step_data['message']}
    </div>
    ''', unsafe_allow_html=True)
    
    # Play audio
    if step_data.get('audio'):
        try:
            with open(step_data['audio'], 'rb') as f:
                st.audio(f.read(), format='audio/mp3', autoplay=True)
        except:
            pass
    
    # Get user response
    next_step = step_data.get('step')
    if next_step == 'complete':
        # Data collection complete, move to assessment
        profile.data_collection_complete = True
        st.session_state.current_step = 'assessment'
        st.rerun()
    else:
        show_response_interface(next_step)

def handle_assessment_step():
    """Handle assessment conversation"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    
    if 'assessment_questions_asked' not in st.session_state:
        st.session_state.assessment_questions_asked = 0
        st.session_state.max_assessment_questions = 5
    
    questions_asked = st.session_state.assessment_questions_asked
    max_questions = st.session_state.max_assessment_questions
    
    if questions_asked < max_questions:
        # Generate next coaching question
        if f'assessment_q_{questions_asked}' not in st.session_state:
            question_context = f"This is question {questions_asked + 1} of {max_questions} about their mental performance"
            question_result = coach.generate_coaching_question(profile, question_context)
            st.session_state[f'assessment_q_{questions_asked}'] = question_result
        
        question_data = st.session_state[f'assessment_q_{questions_asked}']
        
        # Show coach question
        st.markdown(f'''
        <div class="step-indicator">
            Assessment Question {questions_asked + 1} of {max_questions}
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="coach-message">
            <strong>🎤 Coach Alex:</strong> {question_data['question']}
        </div>
        ''', unsafe_allow_html=True)
        
        # Play audio
        if question_data.get('audio'):
            try:
                with open(question_data['audio'], 'rb') as f:
                    st.audio(f.read(), format='audio/mp3', autoplay=True)
            except:
                pass
        
        # Get response and handle next question
        show_assessment_response_interface()
    else:
        # Assessment complete, move to coaching
        st.session_state.current_step = 'coaching'
        st.rerun()

def handle_coaching_step():
    """Handle final coaching recommendations"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    
    if 'final_coaching' not in st.session_state:
        with st.spinner("🎤 Coach Alex is preparing your personalized coaching plan..."):
            final_result = coach.generate_final_coaching_summary(profile, st.session_state.conversation_history)
            st.session_state.final_coaching = final_result
    
    coaching_data = st.session_state.final_coaching
    
    st.markdown(f'''
    <div class="step-indicator">
        🏆 Your Personalized Coaching Plan
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="coach-message">
        <strong>🎤 Coach Alex - Final Recommendations:</strong><br>
        {coaching_data['summary']}
    </div>
    ''', unsafe_allow_html=True)
    
    # Play final coaching audio
    if coaching_data.get('audio'):
        st.subheader("🎧 Listen to Your Coaching Plan")
        try:
            with open(coaching_data['audio'], 'rb') as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')
                
                # Download option
                st.download_button(
                    "💾 Download Your Coaching Audio",
                    audio_bytes,
                    f"coaching_plan_{profile.name}.mp3",
                    "audio/mp3"
                )
        except:
            pass
    
    # Session completion options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 New Session", type="primary", use_container_width=True):
            start_new_session()
    
    with col2:
        if st.button("💾 Save Session", use_container_width=True):
            save_session_data()

def show_response_interface(next_step: str):
    """Show response interface for voice and text input"""
    st.markdown('''
    <div class="voice-controls">
        <h4>🎤 Your Response</h4>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Speak Your Answer", type="primary", use_container_width=True):
            voice_response = get_voice_input("Speak your response:")
            if voice_response:
                add_to_conversation(voice_response, next_step)
    
    with col2:
        # Text input with enter key support
        text_response = st.text_input(
            "Or type your response:",
            key=f"text_input_{next_step}_{len(st.session_state.conversation_history)}",
            placeholder="Type your answer here..."
        )
        
        if text_response:
            if st.button("Send", type="secondary", use_container_width=True):
                add_to_conversation(text_response, next_step)

def show_assessment_response_interface():
    """Show response interface for assessment questions"""
    questions_asked = st.session_state.assessment_questions_asked
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Speak Your Answer", type="primary", use_container_width=True):
            voice_response = get_voice_input("Share your thoughts:")
            if voice_response:
                process_assessment_response(voice_response)
    
    with col2:
        text_response = st.text_input(
            "Or type your response:",
            key=f"assessment_text_{questions_asked}",
            placeholder="Share your thoughts..."
        )
        
        if text_response:
            if st.button("Send Answer", type="secondary", use_container_width=True):
                process_assessment_response(text_response)

def process_assessment_response(user_response: str):
    """Process assessment response and generate coach reply"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    questions_asked = st.session_state.assessment_questions_asked
    
    # Add user response to conversation
    current_question = st.session_state[f'assessment_q_{questions_asked}']['question']
    
    # Generate coach response to their answer
    coach_response = coach.generate_coaching_response(profile, user_response, st.session_state.conversation_history)
    
    # Add to conversation history with consistent structure
    exchange = {
        'coach_message': current_question,  # Fixed: Always use 'coach_message'
        'user_response': user_response,
        'coach_response': coach_response['response'],
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.conversation_history.append(exchange)
    
    # Show user message
    st.markdown(f'''
    <div class="user-message">
        <strong>You:</strong> {user_response}
    </div>
    ''', unsafe_allow_html=True)
    
    # Show coach response
    st.markdown(f'''
    <div class="coach-message">
        <strong>🎤 Coach Alex:</strong> {coach_response['response']}
    </div>
    ''', unsafe_allow_html=True)
    
    # Play coach response audio
    if coach_response.get('audio'):
        try:
            with open(coach_response['audio'], 'rb') as f:
                st.audio(f.read(), format='audio/mp3', autoplay=True)
        except:
            pass
    
    # Move to next question
    st.session_state.assessment_questions_asked += 1
    
    # Small delay then continue
    time.sleep(1)
    st.rerun()

def add_to_conversation(user_response: str, next_step: str):
    """Add user response to conversation and move to next step"""
    
    # Get current coach message with proper fallback handling
    current_message = ""
    if 'welcome_message' in st.session_state:
        current_message = st.session_state.welcome_message['message']
    elif f'step_{st.session_state.current_step}_message' in st.session_state:
        current_message = st.session_state[f'step_{st.session_state.current_step}_message']['message']
    elif st.session_state.conversation_history:
        # Get the last coach message from history
        last_exchange = st.session_state.conversation_history[-1]
        current_message = last_exchange.get('coach_message', 'Coach message')
    else:
        current_message = "Coach message"  # Fallback
    
    # Add to conversation history with consistent structure
    exchange = {
        'coach_message': current_message,  # Fixed: Always use 'coach_message'
        'user_response': user_response,
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.conversation_history.append(exchange)
    
    # Show user message
    st.markdown(f'''
    <div class="user-message">
        <strong>You:</strong> {user_response}
    </div>
    ''', unsafe_allow_html=True)
    
    # Update current step
    st.session_state.current_step = next_step
    
    # Clear step-specific session state
    for key in list(st.session_state.keys()):
        if key.startswith('step_') and key.endswith('_message'):
            del st.session_state[key]
    
    # Small delay for natural conversation flow
    time.sleep(0.5)
    st.rerun()

def get_voice_input(prompt: str) -> Optional[str]:
    """Get voice input from user"""
    coach = st.session_state.voice_coach
    
    st.info(f"🎤 {prompt}")
    
    try:
        with st.spinner("🎤 Listening..."):
            response = coach.voice_service.listen_for_speech(timeout=10)
            
            if response:
                st.success(f"✅ Heard: {response}")
                return response
            else:
                st.warning("No speech detected. Try speaking again or use text input.")
                return None
                
    except Exception as e:
        st.error(f"Voice input error: {e}")
        return None

def start_new_session():
    """Start a completely new session"""
    # Clear all session state except voice coach
    for key in list(st.session_state.keys()):
        if key != 'voice_coach':
            del st.session_state[key]
    
    # Initialize new session
    st.session_state.current_profile = AthleteProfile(session_id=str(uuid.uuid4()))
    st.session_state.conversation_history = []
    st.session_state.current_step = 'welcome'
    
    st.success("🔄 New session started!")
    st.rerun()

def save_session_data():
    """Save current session data"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    
    # Add conversation history to profile
    profile.consultation_history = st.session_state.conversation_history
    
    try:
        coach.db_manager.save_athlete_profile(profile)
        st.success(f"✅ Session saved for {profile.name}!")
        
        # Show session summary
        st.markdown(f'''
        <div class="coach-message">
            <h4>💾 Session Summary</h4>
            <p><strong>Athlete:</strong> {profile.name}</p>
            <p><strong>Sport:</strong> {profile.sport_type.value if profile.sport_type else 'Not specified'}</p>
            <p><strong>Level:</strong> {profile.athlete_level.value if profile.athlete_level else 'Not specified'}</p>
            <p><strong>Conversation Exchanges:</strong> {len(st.session_state.conversation_history)}</p>
            <p><strong>Session Duration:</strong> {(datetime.now() - profile.timestamp).seconds // 60} minutes</p>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Save error: {e}")

def create_daily_reminder_audio():
    """Create a daily reminder audio for the athlete"""
    coach = st.session_state.voice_coach
    profile = st.session_state.current_profile
    
    if not profile.name:
        st.warning("Complete your profile first to generate personalized reminders.")
        return
    
    reminder_text = f"""Hey {profile.name}! Quick mental training reminder. 
    Take 3 deep breaths, visualize one successful moment, and set your intention for today. 
    You've got this champion!"""
    
    with st.spinner("🎤 Creating your daily reminder..."):
        audio_file = coach.generate_voice_audio(reminder_text, "daily_reminder")
        
        if audio_file:
            st.success("✅ Daily reminder created!")
            try:
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/mp3')
                    
                    st.download_button(
                        "💾 Download Daily Reminder",
                        audio_bytes,
                        f"daily_reminder_{profile.name}.mp3",
                        "audio/mp3"
                    )
            except Exception as e:
                st.error(f"Audio error: {e}")

def show_conversation_summary():
    """Show a summary of the conversation"""
    if st.session_state.conversation_history:
        with st.expander("📝 Conversation Summary"):
            for i, exchange in enumerate(st.session_state.conversation_history):
                st.markdown(f"**Exchange {i+1}:**")
                # Fixed: Use consistent key names with fallbacks
                coach_msg = exchange.get('coach_message', exchange.get('question', 'N/A'))
                user_msg = exchange.get('user_response', exchange.get('response', 'N/A'))
                coach_response = exchange.get('coach_response', '')
                
                st.markdown(f"Coach: {coach_msg}")
                st.markdown(f"You: {user_msg}")
                if coach_response:
                    st.markdown(f"Coach Response: {coach_response}")
                st.markdown("---")

def display_athlete_profile_card():
    """Display current athlete profile as a card"""
    profile = st.session_state.current_profile
    
    if profile.name:
        st.markdown(f'''
        <div class="coach-message">
            <h4>👤 {profile.name}'s Profile</h4>
            <p><strong>Age:</strong> {profile.age if profile.age else 'Not set'}</p>
            <p><strong>Sport:</strong> {profile.sport_type.value if profile.sport_type else 'Not set'}</p>
            <p><strong>Level:</strong> {profile.athlete_level.value if profile.athlete_level else 'Not set'}</p>
            <p><strong>Training:</strong> {profile.training_frequency if profile.training_frequency else 'Not set'} times/week</p>
            <p><strong>Goals:</strong> {', '.join(profile.primary_goals) if profile.primary_goals else 'Not set'}</p>
        </div>
        ''', unsafe_allow_html=True)

# Enhanced voice controls
def show_voice_settings():
    """Show voice interaction settings"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔊 Voice Settings")
    
    # Voice input sensitivity
    sensitivity = st.sidebar.slider("Microphone Sensitivity", 0.1, 1.0, 0.5, 0.1)
    if 'voice_coach' in st.session_state:
        st.session_state.voice_coach.voice_service.recognizer.energy_threshold = int(sensitivity * 1000)
    
    # Audio playback speed
    playback_speed = st.sidebar.slider("Playback Speed", 0.8, 1.5, 1.0, 0.1)
    
    # Voice preference
    voice_mode = st.sidebar.radio(
        "Interaction Mode",
        ["Voice + Text", "Voice Only", "Text Only"]
    )
    
    if 'current_profile' in st.session_state:
        st.session_state.current_profile.preferences['voice_mode'] = voice_mode

def test_elevenlabs_connection():
    """Test ElevenLabs API connection"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 ElevenLabs Status")
    
    if Config.ELEVENLABS_API_KEY == "your_elevenlabs_api_key_here":
        st.sidebar.error("❌ ElevenLabs API key not set")
        st.sidebar.markdown("""
        Set your API key:
        ```bash
        export ELEVENLABS_API_KEY="your_key_here"
        ```
        """)
    else:
        if st.sidebar.button("Test ElevenLabs"):
            try:
                coach = st.session_state.voice_coach
                test_audio = coach.voice_service.text_to_speech_elevenlabs("Test message from Coach Alex")
                if test_audio:
                    st.sidebar.success("✅ ElevenLabs working!")
                else:
                    st.sidebar.error("❌ ElevenLabs connection failed")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")

def show_microphone_test():
    """Show microphone test interface"""
    st.sidebar.markdown("---")
    st.sidebar.header("🎤 Microphone Test")
    
    if st.sidebar.button("Test Microphone"):
        if 'voice_coach' in st.session_state:
            coach = st.session_state.voice_coach
            st.sidebar.info("🎤 Say something...")
            test_result = coach.voice_service.listen_for_speech(timeout=5)
            
            if test_result:
                st.sidebar.success(f"✅ Heard: {test_result}")
            else:
                st.sidebar.warning("❌ Microphone test failed")
        else:
            st.sidebar.warning("Voice coach not initialized")

# Error handling and recovery
def handle_conversation_errors():
    """Handle conversation flow errors"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Troubleshooting")
    
    if st.sidebar.button("🔄 Reset Conversation"):
        # Reset conversation but keep profile data
        st.session_state.conversation_history = []
        st.session_state.current_step = 'welcome'
        
        # Clear step messages
        for key in list(st.session_state.keys()):
            if 'step_' in key or 'assessment_' in key or 'welcome_message' in key:
                del st.session_state[key]
        
        st.sidebar.success("Conversation reset!")
        st.rerun()
    
    if st.sidebar.button("🗄️ Reset Database"):
        try:
            if os.path.exists(Config.DATABASE_PATH):
                os.remove(Config.DATABASE_PATH)
            st.sidebar.success("Database reset! Restart the app.")
        except Exception as e:
            st.sidebar.error(f"Database reset error: {e}")

def show_conversation_controls():
    """Show conversation control buttons"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏸️ Pause Session"):
            st.session_state.session_paused = True
            st.info("Session paused. Click 'Resume' to continue.")
    
    with col2:
        if st.session_state.get('session_paused', False):
            if st.button("▶️ Resume Session"):
                st.session_state.session_paused = False
                st.rerun()
    
    with col3:
        if st.button("📊 Show Summary"):
            show_conversation_summary()

def show_debug_info():
    """Show debug information for troubleshooting"""
    if st.sidebar.checkbox("🐛 Debug Mode"):
        st.sidebar.markdown("---")
        st.sidebar.header("Debug Info")
        
        # Current state
        st.sidebar.write(f"**Current Step:** {st.session_state.get('current_step', 'None')}")
        st.sidebar.write(f"**Profile Name:** {st.session_state.current_profile.name}")
        st.sidebar.write(f"**History Length:** {len(st.session_state.conversation_history)}")
        st.sidebar.write(f"**Data Complete:** {st.session_state.current_profile.data_collection_complete}")
        
        # Session state keys
        with st.sidebar.expander("Session State Keys"):
            keys = [k for k in st.session_state.keys() if not k.startswith('_')]
            for key in sorted(keys):
                st.sidebar.write(f"- {key}")
        
        # Show conversation history structure for debugging
        with st.sidebar.expander("Conversation History Structure"):
            for i, exchange in enumerate(st.session_state.conversation_history):
                st.sidebar.write(f"Exchange {i}: {list(exchange.keys())}")

# Database migration utility
def migrate_database():
    """Migrate database to latest schema"""
    try:
        db_manager = DatabaseManager()
        st.sidebar.success("✅ Database migration completed")
    except Exception as e:
        st.sidebar.error(f"❌ Database migration failed: {e}")

# Main application with enhanced features
if __name__ == "__main__":
    
    # System requirements check
    st.markdown("### 🔧 System Setup")
    
    with st.expander("Required Setup Instructions", expanded=False):
        st.markdown("""
        **1. Install ElevenLabs:**
        ```bash
        pip install elevenlabs
        ```
        
        **2. Set ElevenLabs API Key:**
        ```bash
        export ELEVENLABS_API_KEY="your_api_key_here"
        ```
        Get your API key from: https://elevenlabs.io/
        
        **3. Install Ollama and Llama 3.2:**
        ```bash
        curl -fsSL https://ollama.ai/install.sh | sh
        ollama pull llama3.2:latest
        ollama serve
        ```
        
        **4. Install Voice Libraries:**
        ```bash
        pip install SpeechRecognition pyaudio pygame
        ```
        
        **Voice Features:**
        - High-quality ElevenLabs TTS with Adam (male voice)
        - Conversational data collection
        - Short, chatbot-style responses
        - Natural voice interaction flow
        - Confident male coach personality
        """)
    
    # System status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Ollama status
        try:
            response = requests.get(f"{Config.API_BASE}/api/tags", timeout=2)
            st.success("✅ Ollama") if response.status_code == 200 else st.error("❌ Ollama")
        except:
            st.error("❌ Ollama")
    
    with col2:
        # ElevenLabs status
        if Config.ELEVENLABS_API_KEY != "your_elevenlabs_api_key_here":
            st.success("✅ ElevenLabs")
        else:
            st.warning("⚠️ ElevenLabs")
    
    with col3:
        # Speech recognition status
        try:
            import speech_recognition
            st.success("✅ Voice Input")
        except ImportError:
            st.error("❌ Voice Input")
    
    with col4:
        # Database status
        try:
            db_manager = DatabaseManager()
            st.success("✅ Database")
        except Exception as e:
            st.error("❌ Database")
            if st.button("Fix Database"):
                migrate_database()
    
    # Initialize and run main app
    main()
    
    # Sidebar enhancements
    show_voice_settings()
    test_elevenlabs_connection()
    show_microphone_test()
    handle_conversation_errors()
    show_debug_info()
    
    # Additional features in main area
    if st.session_state.get('current_profile') and st.session_state.current_profile.name:
        
        st.markdown("---")
        st.subheader("🎯 Session Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📱 Create Daily Reminder"):
                create_daily_reminder_audio()
        
        with col2:
            if st.button("📝 View Profile"):
                display_athlete_profile_card()
        
        with col3:
            if st.button("📊 Conversation Log"):
                show_conversation_summary()
    
    # Footer with usage tips
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        💡 <strong>Tips:</strong> Speak naturally • Use headphones for best experience • 
        Test your microphone first • Each conversation is saved automatically • 
        🎤 Now featuring Adam - confident male coaching voice
    </div>
    """, unsafe_allow_html=True)
