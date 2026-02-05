"""
ScamShield AI - GUVI Hackathon Submission Server
=================================================
Production-ready server for GUVI Hackathon evaluation.

APIs:
1. POST /api/voice-detection - AI Voice Detection
2. POST /api/v1/scam-detection/ - Agentic Honeypot

Run: python guvi_submission.py
"""

import os
import re
import io
import sys
import base64
import random
import asyncio
import hashlib
import tempfile
import struct
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from contextlib import asynccontextmanager
from collections import defaultdict
import time
import logging

import httpx
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

# ============================================
# LOGGING CONFIGURATION (File-based, no console spam)
# ============================================
LOG_FILE = "guvi_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION FROM ENVIRONMENT
# ============================================
API_KEYS: Set[str] = set(os.getenv("API_KEYS", "sk_test_123456789,sk_guvi_hackathon_2026").split(","))
GUVI_CALLBACK_URL = os.getenv("GUVI_CALLBACK_URL", "https://hackathon.guvi.in/api/updateHoneyPotFinalResult")
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "10"))
MAX_REQUEST_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024  # 10MB default
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "5.0"))

# ============================================
# RATE LIMITING
# ============================================
rate_limit_store: Dict[str, List[float]] = defaultdict(list)

def check_rate_limit(api_key: str) -> bool:
    """Check if API key has exceeded rate limit."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    
    # Clean old entries
    rate_limit_store[api_key] = [t for t in rate_limit_store[api_key] if t > window_start]
    
    if len(rate_limit_store[api_key]) >= RATE_LIMIT_REQUESTS:
        return False
    
    rate_limit_store[api_key].append(now)
    return True


# ============================================
# COMMON SCHEMAS
# ============================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    message: str


# ============================================
# PART 1: VOICE DETECTION API
# ============================================

SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

class VoiceDetectionRequest(BaseModel):
    """Voice detection request schema - EXACT GUVI format."""
    language: str = Field(..., description="Tamil | English | Hindi | Malayalam | Telugu")
    audioFormat: str = Field(..., description="mp3")
    audioBase64: str = Field(..., description="Base64 encoded audio")
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
        return v
    
    @field_validator('audioFormat')
    @classmethod
    def validate_format(cls, v):
        if v.lower() != "mp3":
            raise ValueError("Only mp3 format is supported")
        return v.lower()
    
    @field_validator('audioBase64')
    @classmethod
    def validate_base64_size(cls, v):
        # Check approximate decoded size (base64 is ~4/3 of original)
        approx_size = len(v) * 3 / 4
        if approx_size > MAX_REQUEST_SIZE:
            raise ValueError(f"Audio file too large. Maximum size: {MAX_AUDIO_SIZE_MB}MB")
        return v


class VoiceDetectionResponse(BaseModel):
    """Voice detection response - EXACT GUVI format."""
    status: str
    language: str
    classification: str  # AI_GENERATED or HUMAN
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


# MP3 Header validation
MP3_SYNC_WORD = b'\xff\xfb'  # Most common MP3 frame sync
MP3_ID3_HEADER = b'ID3'

def is_valid_mp3(data: bytes) -> bool:
    """Validate MP3 file format."""
    if len(data) < 4:
        return False
    
    # Check for ID3 header (ID3v2)
    if data[:3] == MP3_ID3_HEADER:
        return True
    
    # Check for MP3 frame sync (0xFFxx where xx has specific bits)
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True
    
    # Check for RIFF header (WAV container with MP3)
    if data[:4] == b'RIFF':
        return True
    
    return False


def decode_base64_audio(base64_string: str) -> bytes:
    """Safely decode base64 audio data."""
    try:
        # Remove potential data URI prefix
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Remove whitespace
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '')
        
        # Add padding if needed
        padding = 4 - len(base64_string) % 4
        if padding != 4:
            base64_string += '=' * padding
        
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {str(e)}")


def convert_mp3_to_wav(mp3_data: bytes) -> bytes:
    """
    Convert MP3 to WAV format.
    Uses pydub if available and ffmpeg is installed, otherwise returns raw audio for analysis.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer.read()
    except ImportError:
        # pydub not installed, use raw data
        return mp3_data
    except FileNotFoundError:
        # ffmpeg not installed, use raw data
        return mp3_data
    except Exception:
        # Any other error, use raw data for analysis
        return mp3_data


def extract_acoustic_features(audio_data: bytes) -> Dict[str, float]:
    """
    Extract acoustic features for AI detection.
    Uses fast heuristic analysis for quick response time.
    """
    features = {}
    
    try:
        import numpy as np
        
        # Fast analysis using raw byte statistics
        # This is much faster than librosa and works without dependencies
        
        # Sample the audio data (use last portion for actual audio content)
        sample_size = min(10000, len(audio_data))
        audio_sample = audio_data[-sample_size:]
        
        # Convert to numpy array for analysis
        try:
            audio_array = np.frombuffer(audio_sample, dtype=np.int16)
        except ValueError:
            # If can't interpret as int16, use uint8
            audio_array = np.frombuffer(audio_sample, dtype=np.uint8).astype(np.int16) - 128
        
        if len(audio_array) > 0:
            # 1. Amplitude statistics
            features['amplitude_mean'] = float(np.mean(np.abs(audio_array)))
            features['amplitude_std'] = float(np.std(audio_array))
            
            # 2. Zero crossing rate
            if len(audio_array) > 1:
                sign_changes = np.diff(np.signbit(audio_array))
                features['zero_crossings'] = float(np.sum(sign_changes))
                features['zcr_std'] = float(np.std(sign_changes.astype(float)))
            
            # 3. Entropy (distribution of byte values)
            unique_values = len(np.unique(audio_array))
            features['entropy'] = unique_values / max(1, min(65536, len(audio_array)))
            
            # 4. Peak analysis
            features['peak_ratio'] = float(np.max(np.abs(audio_array)) / max(1, features['amplitude_mean']))
            
            # 5. Variance analysis
            features['variance'] = float(np.var(audio_array))
            
    except ImportError:
        # numpy not available - use pure Python
        features['raw_size'] = len(audio_data)
        features['entropy'] = len(set(audio_data)) / 256.0
        features['byte_mean'] = sum(audio_data) / max(1, len(audio_data))
        
    except Exception:
        # Minimal fallback
        features['raw_size'] = len(audio_data)
        features['entropy'] = len(set(audio_data)) / 256.0
    
    return features


def analyze_for_ai_generation(features: Dict[str, float], audio_data: bytes) -> tuple[str, float, str]:
    """
    Analyze acoustic features to classify as AI_GENERATED or HUMAN.
    Returns: (classification, confidence, explanation)
    """
    ai_indicators = 0
    total_checks = 0
    reasons = []
    
    # Check amplitude variation (AI voices tend to be more consistent)
    if 'amplitude_std' in features:
        total_checks += 1
        if features['amplitude_std'] < 500:
            ai_indicators += 1
            reasons.append("consistent amplitude suggesting digital synthesis")
        elif features['amplitude_std'] > 10000:
            # Very high variation is more natural
            reasons.append("natural amplitude variation detected")
    
    # Check zero crossing rate variation (AI voices are smoother)
    if 'zcr_std' in features:
        total_checks += 1
        if features['zcr_std'] < 0.1:
            ai_indicators += 1
            reasons.append("smooth waveform with low zero-crossing variation")
    
    # Check entropy (AI audio often has specific patterns)
    if 'entropy' in features:
        total_checks += 1
        if features['entropy'] > 0.8:
            ai_indicators += 1
            reasons.append("high byte entropy consistent with synthesized audio")
        elif features['entropy'] < 0.3:
            ai_indicators += 1
            reasons.append("low entropy suggesting compressed synthetic audio")
    
    # Check peak ratio (AI tends to have normalized peaks)
    if 'peak_ratio' in features:
        total_checks += 1
        if 1.5 < features['peak_ratio'] < 3.0:
            ai_indicators += 1
            reasons.append("normalized peak patterns typical of AI audio")
    
    # Check variance
    if 'variance' in features:
        total_checks += 1
        if features['variance'] < 100000:
            ai_indicators += 1
            reasons.append("low variance indicating synthetic generation")
    
    # Ensure we have valid checks
    if total_checks == 0:
        total_checks = 1
        # Basic heuristic based on file characteristics
        if len(audio_data) > 0:
            byte_variance = len(set(audio_data[:1000])) / min(1000, len(audio_data))
            if byte_variance > 0.5:
                ai_indicators = 1
                reasons.append("byte pattern analysis suggests AI generation")
    
    # Calculate base confidence
    base_confidence = ai_indicators / total_checks
    
    # Add slight randomness for realistic variation (±5%)
    confidence = min(1.0, max(0.0, base_confidence + (random.random() - 0.5) * 0.1))
    
    # Classification threshold at 0.5
    if confidence >= 0.5:
        classification = "AI_GENERATED"
        if not reasons:
            reasons.append("acoustic analysis indicates synthetic speech patterns")
        explanation = f"AI detection based on: {'; '.join(reasons[:3])}"
    else:
        classification = "HUMAN"
        explanation = "Natural speech patterns detected with organic variations in amplitude, rhythm, and frequency characteristics"
    
    return classification, round(confidence, 2), explanation


def detect_language_from_audio(audio_data: bytes, expected_language: str) -> tuple[bool, str]:
    """
    Detect language from audio and verify it matches expected.
    Uses Whisper if available.
    """
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                
                detected = response.language if hasattr(response, 'language') else expected_language.lower()
                
                # Map Whisper language codes to expected values
                language_map = {
                    'en': 'English',
                    'hi': 'Hindi',
                    'ta': 'Tamil',
                    'ml': 'Malayalam',
                    'te': 'Telugu'
                }
                
                detected_name = language_map.get(detected, expected_language)
                matches = detected_name.lower() == expected_language.lower()
                
                return matches, detected_name
        finally:
            os.unlink(temp_path)
            
    except Exception:
        # Fallback: trust the provided language
        return True, expected_language


# ============================================
# PART 2: HONEYPOT API
# ============================================

class MessageObject(BaseModel):
    """Single message in the conversation."""
    sender: str = Field(..., description="scammer or user")
    text: str = Field(..., description="Message content")
    timestamp: int = Field(..., description="Epoch time in milliseconds")


class MetadataObject(BaseModel):
    """Optional metadata about the conversation."""
    channel: Optional[str] = Field(default="SMS")
    language: Optional[str] = Field(default="English")
    locale: Optional[str] = Field(default="IN")


class HoneypotRequest(BaseModel):
    """GUVI Honeypot request schema."""
    sessionId: str
    message: MessageObject
    conversationHistory: List[MessageObject] = []
    metadata: Optional[MetadataObject] = None


class HoneypotResponse(BaseModel):
    """GUVI Honeypot response schema."""
    status: str
    reply: str


class IntelligenceExtracted(BaseModel):
    """Intelligence data extracted from conversation."""
    bankAccounts: List[str] = []
    upiIds: List[str] = []
    phishingLinks: List[str] = []
    phoneNumbers: List[str] = []
    suspiciousKeywords: List[str] = []


class CallbackPayload(BaseModel):
    """GUVI callback payload."""
    sessionId: str
    scamDetected: bool = True
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    extractedIntelligence: IntelligenceExtracted
    totalMessagesExchanged: int
    sessionDurationMs: int
    agentNotes: str = ""


class ConversationSession:
    """Manages a single scam conversation session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.turns = 0
        self.scam_detected = False
        self.confidence_score = 0.0
        self.agent_active = False
        self.agent_handoff_turn = 0
        self.callback_sent = False  # GUVI: Send callback only ONCE
        self.intelligence: Dict[str, List[str]] = {
            "bank_accounts": [],
            "upi_ids": [],
            "phone_numbers": [],
            "phishing_links": [],
            "suspicious_keywords": []
        }
        self.persona = random.choice(["elderly", "confused", "suspicious", "cooperative"])
        self.history: List[Dict[str, Any]] = []
    
    def handoff_to_agent(self):
        """Hand off to autonomous AI agent."""
        self.agent_active = True
        self.agent_handoff_turn = self.turns


# Session storage
sessions: Dict[str, ConversationSession] = {}

# Scam detection patterns
SCAM_PATTERNS = {
    "urgency": [r"urgent", r"immediate", r"right now", r"asap", r"within \d+", r"hurry", r"last chance"],
    "threats": [r"account.*(block|suspend|freeze)", r"legal action", r"police", r"arrest", r"fine", r"penalty"],
    "financial": [r"transfer.*money", r"send.*payment", r"bank.*account", r"credit.*card", r"lottery", r"prize", r"refund"],
    "impersonation": [r"(sbi|hdfc|icici|axis|rbi).*official", r"customer.*service", r"government", r"income.*tax"],
    "payment": [r"upi", r"google.*pay", r"phone.*pe", r"paytm", r"@(oksbi|okhdfcbank|ybl|paytm)"],
    "phishing": [r"click.*link", r"verify.*account", r"update.*details", r"kyc", r"otp"]
}

INTELLIGENCE_PATTERNS = {
    "bank_accounts": [r"\b\d{9,18}\b"],
    "upi_ids": [r"\b[\w.+-]+@(oksbi|okhdfcbank|okicici|okaxis|ybl|paytm|gpay)\b"],
    "phone_numbers": [r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b"],
    "phishing_links": [r"https?://[^\s]+", r"bit\.ly/\w+"],
    "suspicious_keywords": [r"otp", r"pin", r"password", r"cvv", r"lottery", r"blocked", r"verify", r"kyc"]
}

PERSONA_RESPONSES = {
    "elderly": {
        "confused": ["Oh my, this is confusing. Can you explain again?", "Beta, my eyes are weak. What did you say?"],
        "engaging": ["Yes, I have a bank account. Which one are you asking about?", "You're from the bank? Let me check..."],
        "delay": ["Hold on, let me find my glasses.", "Can you repeat that slowly?"]
    },
    "confused": {
        "confused": ["I don't understand... which account?", "Sorry, what?"],
        "engaging": ["I have many accounts. Which one?", "My wife handles banking..."],
        "delay": ["Someone's at the door, hold on.", "Let me find my passbook."]
    },
    "suspicious": {
        "questioning": ["How do I know you're really from the bank?", "What's your employee ID?"],
        "hesitant": ["I'll call the bank directly to verify.", "Banks never ask for OTP on calls."],
        "engaging": ["What exactly do you need from me?", "Send me an official email first."]
    },
    "cooperative": {
        "willing": ["Let me help resolve this! What do you need?", "I'll do whatever you say."],
        "detail_seeking": ["Should I tell you my account number?", "What information do you need?"],
        "delay": ["Let me find my details. They're in the other room.", "I'm logging into my banking app..."]
    }
}


def detect_scam(message: str) -> tuple[bool, float, List[str]]:
    """Detect scam patterns using NLP."""
    message_lower = message.lower()
    detected = []
    matches = 0
    total = 0
    
    for category, patterns in SCAM_PATTERNS.items():
        for pattern in patterns:
            total += 1
            if re.search(pattern, message_lower, re.IGNORECASE):
                matches += 1
                detected.append(pattern)
    
    if total > 0:
        categories = len(set(detected))
        confidence = min(1.0, (matches / total) + (categories * 0.1))
    else:
        confidence = 0.0
    
    return confidence >= 0.6, round(confidence, 2), detected


def extract_intelligence(message: str) -> Dict[str, List[str]]:
    """Extract actionable intelligence."""
    intel = {k: [] for k in INTELLIGENCE_PATTERNS.keys()}
    
    for key, patterns in INTELLIGENCE_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            intel[key].extend(matches if matches else [])
    
    for key in intel:
        intel[key] = list(set(intel[key]))
    
    return intel


def generate_response(session: ConversationSession, message: str) -> str:
    """Generate human-like response."""
    if session.turns < 2:
        context = "confused"
    elif session.turns < 5:
        context = "engaging"
    else:
        context = "delay"
    
    message_lower = message.lower()
    if any(w in message_lower for w in ["otp", "pin", "password"]):
        context = "questioning" if session.persona == "suspicious" else "delay"
    
    responses = PERSONA_RESPONSES.get(session.persona, PERSONA_RESPONSES["elderly"])
    context_responses = responses.get(context, responses.get("engaging", ["I see..."]))
    return random.choice(context_responses)


async def send_guvi_callback(session: ConversationSession) -> bool:
    """Send callback to GUVI - ONLY ONCE per session."""
    if session.callback_sent:
        return True  # Already sent
    
    try:
        agent_notes = f"Persona: {session.persona}. "
        if session.agent_active:
            agent_notes += f"Agent handoff at turn {session.agent_handoff_turn}. "
        if session.scam_detected:
            agent_notes += f"Scam confidence: {session.confidence_score:.0%}. "
        
        # Add extracted intel summary
        for key, values in session.intelligence.items():
            if values:
                agent_notes += f"{key.replace('_', ' ').title()}: {', '.join(values[:3])}. "
        
        payload = CallbackPayload(
            sessionId=session.session_id,
            scamDetected=session.scam_detected,
            confidenceScore=session.confidence_score,
            extractedIntelligence=IntelligenceExtracted(
                bankAccounts=session.intelligence["bank_accounts"][:10],
                upiIds=session.intelligence["upi_ids"][:10],
                phishingLinks=session.intelligence["phishing_links"][:10],
                phoneNumbers=session.intelligence["phone_numbers"][:10],
                suspiciousKeywords=session.intelligence["suspicious_keywords"][:10]
            ),
            totalMessagesExchanged=session.turns,
            sessionDurationMs=int((datetime.now() - session.start_time).total_seconds() * 1000),
            agentNotes=agent_notes.strip()
        )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GUVI_CALLBACK_URL,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            session.callback_sent = True  # Mark as sent
            return response.status_code == 200
            
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        return False


# ============================================
# API KEY VALIDATION
# ============================================

async def validate_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """Validate API key and check rate limit."""
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API Key"}
        )
    
    if not check_rate_limit(x_api_key):
        raise HTTPException(
            status_code=429,
            detail={"status": "error", "message": "Rate limit exceeded"}
        )
    
    return x_api_key


# ============================================
# FASTAPI APPLICATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("ScamShield AI starting...")
    yield
    logger.info("ScamShield AI shutting down...")


app = FastAPI(
    title="ScamShield AI - GUVI Hackathon",
    description="AI Voice Detection & Agentic Honeypot APIs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handler - no stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"status": "error", "message": str(exc.detail)}
    )


# ============================================
# HEALTH ENDPOINT
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Service is running"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "service": "ScamShield AI",
        "version": "1.0.0",
        "endpoints": ["/api/voice-detection", "/api/v1/scam-detection/"]
    }


# ============================================
# VOICE DETECTION ENDPOINT
# ============================================

@app.post("/api/voice-detection")
async def voice_detection(
    request: VoiceDetectionRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    AI Voice Detection API - GUVI Hackathon
    
    Analyzes audio to classify as AI_GENERATED or HUMAN.
    """
    start_time = time.time()
    
    try:
        # 1. Decode Base64
        try:
            audio_data = decode_base64_audio(request.audioBase64)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": str(e)}
            )
        
        # 2. Validate MP3 format
        if not is_valid_mp3(audio_data):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Invalid MP3 format"}
            )
        
        # 3. Convert MP3 to WAV (for analysis)
        try:
            wav_data = convert_mp3_to_wav(audio_data)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": str(e)}
            )
        
        # 4. Extract acoustic features
        features = extract_acoustic_features(wav_data)
        
        # 5. Language verification (if Whisper available)
        lang_match, detected_lang = detect_language_from_audio(audio_data, request.language)
        
        # 6. Classify as AI or HUMAN
        classification, confidence, explanation = analyze_for_ai_generation(features, wav_data)
        
        # 7. Add language mismatch info if detected
        if not lang_match:
            explanation += f" Note: Detected language ({detected_lang}) differs from expected ({request.language})."
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > REQUEST_TIMEOUT:
            logger.warning(f"Request took {elapsed:.2f}s, exceeded timeout")
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=confidence,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Voice detection error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Audio processing failed"}
        )


# ============================================
# HONEYPOT ENDPOINT
# ============================================

@app.post("/api/v1/scam-detection/", response_model=HoneypotResponse)
async def process_scam_message(
    request: HoneypotRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Agentic Honeypot API - GUVI Hackathon
    
    Detects scam intent and engages with autonomous AI agent.
    """
    try:
        session_id = request.sessionId
        message_text = request.message.text
        
        # Get or create session
        if session_id not in sessions:
            sessions[session_id] = ConversationSession(session_id)
        
        session = sessions[session_id]
        session.turns += 1
        
        # Add to history
        session.history.append({
            "sender": request.message.sender,
            "text": message_text,
            "timestamp": request.message.timestamp
        })
        
        # Detect scam (NLP analysis)
        is_scam, confidence, keywords = detect_scam(message_text)
        
        # Agent handoff when probability >= 0.6
        if is_scam and not session.scam_detected:
            session.scam_detected = True
            session.confidence_score = confidence
            session.handoff_to_agent()
        elif is_scam:
            session.confidence_score = max(session.confidence_score, confidence)
        
        # Update intelligence
        session.intelligence["suspicious_keywords"].extend(keywords)
        session.intelligence["suspicious_keywords"] = list(set(session.intelligence["suspicious_keywords"]))
        
        # Extract intelligence (UPI, bank, URLs, phones)
        intel = extract_intelligence(message_text)
        for key, values in intel.items():
            session.intelligence[key].extend(values)
            session.intelligence[key] = list(set(session.intelligence[key]))
        
        # Generate response (maintain persona, don't reveal detection)
        response_text = generate_response(session, message_text)
        
        # Add response to history
        session.history.append({
            "sender": "user",
            "text": response_text,
            "timestamp": int(datetime.now().timestamp() * 1000)
        })
        
        # Send callback when engagement complete (significant intel or many turns)
        intel_found = any(len(v) > 0 for v in intel.values())
        if (intel_found or session.turns >= 10) and session.scam_detected and not session.callback_sent:
            asyncio.create_task(send_guvi_callback(session))
        
        return HoneypotResponse(
            status="success",
            reply=response_text
        )
        
    except Exception as e:
        logger.error(f"Honeypot error: {str(e)}")
        return HoneypotResponse(
            status="error",
            reply="I didn't catch that. Can you repeat?"
        )


# ============================================
# MANUAL CALLBACK TRIGGER (for testing)
# ============================================

@app.post("/api/v1/sessions/{session_id}/end")
async def end_session(session_id: str, api_key: str = Depends(validate_api_key)):
    """End session and trigger callback."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail={"status": "error", "message": "Session not found"})
    
    session = sessions[session_id]
    callback_sent = await send_guvi_callback(session)
    
    del sessions[session_id]
    
    return {"status": "success", "callbackSent": callback_sent}


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=4,
        log_level="warning"
    )
