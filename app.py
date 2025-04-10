# app.py
from flask import Flask, render_template, request, jsonify
import json
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from elevenlabs import set_api_key, generate
from deepgram import Deepgram
from langdetect import detect
import base64
import asyncio
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
# Updated to use eventlet for better asynchronous handling
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
load_dotenv()

# Global config
config = {
    "personA_target_language": "English",
    "personB_target_language": "German",
    "voiceA": "George",
    "voiceB": "Arnold",
    "session_active": False
}

available_voices = {
    "English": ["George", "Rachel", "Emily"],
    "Italian": ["Antonio", "Isabella"],
    "German": ["Arnold", "Klaus"],
    "French": ["Pierre", "Sophie"],
    "Spanish": ["Carlos", "Elena"],
    "Japanese": ["Hiroshi", "Yuki"]
}

available_languages = [
    "English", "Italian", "German", "French", 
    "Spanish", "Japanese", "Portuguese", "Russian", 
    "Chinese", "Arabic", "Hindi", "Dutch"
]

# === GPT setup ===
translation_template = """
Translate the following sentence into {language}, return ONLY the translation, nothing else.

Sentence: {sentence}
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

output_parser = StrOutputParser()
llm = ChatOpenAI(temperature=0.0, model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY) 
translation_prompt = ChatPromptTemplate.from_template(translation_template)

translation_chain = (
    {"language": RunnablePassthrough(), "sentence": RunnablePassthrough()} 
    | translation_prompt
    | llm
    | output_parser
)

def translate(sentence, language):
    data_input = {"language": language, "sentence": sentence}
    return translation_chain.invoke(data_input)

# ElevenLabs
set_api_key(os.getenv("ELEVEN_API_KEY"))

def gen_dub(text, voice="George"):
    audio = generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    return audio

# Deepgram
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = Deepgram(DEEPGRAM_API_KEY)

translation_history = []
mic_handlers = {'A': None, 'B': None}

# Updated MicrophoneHandler class with improved WebSocket handling
class MicrophoneHandler:
    def __init__(self, persona, target_language, voice):
        self.persona = persona
        self.target_language = target_language
        self.voice = voice
        self.active = False
        self.websocket = None
        self.task = None
        self.loop = None
    
    def run_async(self):
        """Run the async loop in its own thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start())
    
    async def start(self):
        self.active = True
        print(f"[Mic {self.persona}] Starting microphone handler...")

        try:
            # Configure Deepgram options
            deepgram_options = {
                'encoding': 'linear16',
                'sample_rate': 44100,
                'channels': 1,
                'language': 'auto',
                'model': 'nova-2',
                'smart_format': True,
                'interim_results': True
            }
            
            # Connect to Deepgram's API
            self.websocket = await deepgram.transcription.live(deepgram_options)
            
            print(f"[Mic {self.persona}] WebSocket connection established")
            
            # Set up event handlers for Deepgram WebSocket
            self.websocket.registerHandler(self.websocket.EVENT_CLOSE, self._on_close)
            self.websocket.registerHandler(self.websocket.EVENT_ERROR, self._on_error)
            self.websocket.registerHandler(self.websocket.EVENT_TRANSCRIPT_RECEIVED, self._on_transcript)

            # Keep the connection alive until stopped
            while self.active:
                await asyncio.sleep(0.5)
                
        except Exception as e:
            print(f"[Mic {self.persona}] ❌ ERROR establishing connection: {str(e)}")
            self.active = False
            
            # Try to reconnect
            if self.active:
                await asyncio.sleep(2)
                await self.start()

    async def _on_close(self):
        """Handle WebSocket close event"""
        print(f"[Mic {self.persona}] WebSocket closed")
        if self.active:
            await asyncio.sleep(2)  # Wait before reconnecting
            await self.start()

    async def _on_error(self, error):
        """Handle WebSocket error event"""
        print(f"[Mic {self.persona}] WebSocket error: {str(error)}")
        # Errors are usually followed by connection close, so we'll reconnect in the _on_close handler

    async def _on_transcript(self, transcript):
        """Process transcript data from Deepgram"""
        try:
            if 'channel' in transcript and 'alternatives' in transcript['channel'] and transcript['channel']['alternatives']:
                transcript_text = transcript['channel']['alternatives'][0]['transcript']
                
                if transcript_text:
                    try:
                        detected_lang = detect(transcript_text)
                    except:
                        detected_lang = "unknown"

                    print(f"[Mic {self.persona}] Detected text: {transcript_text}")
                    
                    # Translate the text
                    translation = translate(transcript_text, self.target_language)
                    print(f"[Mic {self.persona}] Translated to: {translation}")
                    
                    # Generate audio for the translation
                    audio = gen_dub(translation, voice=self.voice)
                    audio_b64 = base64.b64encode(audio).decode('utf-8')

                    # Create a new entry for the translation history
                    new_entry = {
                        "timestamp": time.time(),
                        "persona": self.persona,
                        "original_text": transcript_text,
                        "detected_language": detected_lang,
                        "translated_text": translation,
                        "target_language": self.target_language,
                        "voice": self.voice,
                        "audio_b64": audio_b64
                    }
                    
                    # Add to history and emit to clients
                    translation_history.append(new_entry)
                    if len(translation_history) > 50:
                        translation_history.pop(0)
                    
                    socketio.emit('new_translation', new_entry)
        except Exception as e:
            print(f"[Mic {self.persona}] Error processing transcript: {str(e)}")

    async def send_audio(self, audio_data):
        """Send audio data to Deepgram"""
        if self.websocket and self.active:
            try:
                await self.websocket.send(audio_data)
            except Exception as e:
                print(f"[Mic {self.persona}] Error sending audio: {str(e)}")

    async def stop(self):
        """Stop the microphone handler gracefully"""
        print(f"[Mic {self.persona}] Stopping microphone handler...")
        self.active = False
        
        if self.websocket:
            try:
                await self.websocket.finish()
                print(f"[Mic {self.persona}] WebSocket closed")
            except Exception as e:
                print(f"[Mic {self.persona}] Error closing WebSocket: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html',
                          available_languages=available_languages,
                          available_voices=available_voices,
                          config=config)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    global config
    if request.method == 'POST':
        data = request.json
        for key, value in data.items():
            if key in config:
                config[key] = value
        return jsonify({"status": "success", "config": config})
    else:
        return jsonify(config)

# Updated session management functions
@app.route('/api/start_session', methods=['POST'])
def start_session():
    global config, mic_handlers
    
    if config['session_active']:
        return jsonify({"status": "error", "message": "Session already active"})

    # Clear translation history
    translation_history.clear()
    
    try:
        # Create the microphone handlers
        mic_handlers['A'] = MicrophoneHandler(
            persona='A',
            target_language=config['personB_target_language'],
            voice=config['voiceB']
        )
        
        mic_handlers['B'] = MicrophoneHandler(
            persona='B',
            target_language=config['personA_target_language'],
            voice=config['voiceA']
        )
        
        # Start the handlers in their own threads
        for persona in ['A', 'B']:
            thread = threading.Thread(
                target=mic_handlers[persona].run_async, 
                daemon=True
            )
            thread.start()
        
        config['session_active'] = True
        return jsonify({"status": "success", "message": "Translation session started"})

    except Exception as e:
        print(f"Error starting session: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    global config, mic_handlers
    
    if not config['session_active']:
        return jsonify({"status": "error", "message": "No active session"})

    try:
        # Stop the microphone handlers
        for persona in ['A', 'B']:
            if mic_handlers[persona] and mic_handlers[persona].loop:
                # Schedule stop in the handler's own event loop
                asyncio.run_coroutine_threadsafe(
                    mic_handlers[persona].stop(), 
                    mic_handlers[persona].loop
                )
        
        # Reset handlers
        mic_handlers = {'A': None, 'B': None}
        config['session_active'] = False
        
        return jsonify({"status": "success", "message": "Translation session stopped"})

    except Exception as e:
        print(f"Error stopping session: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/translation_history')
def get_history():
    since = request.args.get('since', 0, type=float)
    filtered_history = [entry for entry in translation_history if entry['timestamp'] > since]
    return jsonify(filtered_history)

@app.route('/api/available_languages')
def get_languages():
    return jsonify(available_languages)

@app.route('/api/available_voices')
def get_voices():
    return jsonify(available_voices)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# Updated audio handler for Socket.IO
@socketio.on('audio')
def handle_audio(data):
    """Handle audio data received from the client"""
    global mic_handlers, config
    
    if not config['session_active']:
        return
    
    try:
        # For simplicity, we're routing all audio through persona A's handler
        # You could add logic to determine which persona based on client ID
        persona = 'A'
        
        if mic_handlers[persona] and mic_handlers[persona].active:
            # Schedule the audio data to be sent to Deepgram
            asyncio.run_coroutine_threadsafe(
                mic_handlers[persona].send_audio(data), 
                mic_handlers[persona].loop
            )
    except Exception as e:
        print(f"Error handling audio from client: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
