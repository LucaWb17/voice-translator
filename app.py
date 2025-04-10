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
# socketio = SocketIO(app, cors_allowed_origins="*")  # Allow connections from any origin
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
# Create a global event loop for asyncio tasks
loop = asyncio.new_event_loop()
thread = None

class MicrophoneHandler:
    def __init__(self, persona, target_language, voice, loop=None):
        self.persona = persona
        self.target_language = target_language
        self.voice = voice
        self.active = False
        self.websocket = None
        self.loop = loop or asyncio.new_event_loop()
    
    def run_async(self):
        """Run the async loop in its own thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start())

# Update in app.py
async def start(self):
    self.active = True
    print(f"[Mic {self.persona}] Starting microphone handler...")

    try:
        # Updated Deepgram connection approach
        deepgram_options = {
            'punctuate': True,
            'language': 'auto',
            'model': 'nova-2',
            'smart_format': True,
            'interim_results': True
        }
        
        # Connect to Deepgram's API with proper error handling
        self.websocket = await deepgram.transcription.live(deepgram_options)
        
        print(f"[Mic {self.persona}] WebSocket connection established")
        
        # Register event handlers for the Deepgram WebSocket
        self.websocket.registerHandler(self.websocket.EVENT_CLOSE, self._on_close)
        self.websocket.registerHandler(self.websocket.EVENT_ERROR, self._on_error)
        self.websocket.registerHandler(self.websocket.EVENT_TRANSCRIPT_RECEIVED, self._on_transcript)
        
        # Keep connection alive
        while self.active:
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"[Mic {self.persona}] ❌ ERROR establishing connection: {str(e)}")
        self.active = False

# Add these handler methods to the MicrophoneHandler class
async def _on_close(self):
    print(f"[Mic {self.persona}] WebSocket closed")
    if self.active:
        await self.restart()

async def _on_error(self, error):
    print(f"[Mic {self.persona}] WebSocket error: {error}")
    if self.active:
        await self.restart()

async def _on_transcript(self, transcript):
    try:
        # Process the transcript
        if transcript['channel']['alternatives'][0]['transcript']:
            text = transcript['channel']['alternatives'][0]['transcript']
            
            try:
                detected_lang = detect(text)
            except:
                detected_lang = "unknown"

            print(f"[Mic {self.persona}] Detected text: {text}")
            
            translation = translate(text, self.target_language)
            print(f"[Mic {self.persona}] Translated to: {translation}")
            
            audio = gen_dub(translation, voice=self.voice)
            audio_b64 = base64.b64encode(audio).decode('utf-8')

            new_entry = {
                "timestamp": time.time(),
                "persona": self.persona,
                "original_text": text,
                "detected_language": detected_lang,
                "translated_text": translation,
                "target_language": self.target_language,
                "voice": self.voice,
                "audio_b64": audio_b64
            }
            
            translation_history.append(new_entry)
            if len(translation_history) > 50:
                translation_history.pop(0)
            
            # Emit the new translation to connected clients
            socketio.emit('new_translation', new_entry)
    except Exception as e:
        print(f"[Mic {self.persona}] Error processing transcript: {str(e)}")

            # Start the transcription handler
            self.task = asyncio.create_task(handle_transcription())
            print(f"[Mic {self.persona}] Transcription task started")

        except Exception as e:
            print(f"[Mic {self.persona}] ❌ ERROR establishing connection: {str(e)}")
            self.active = False

    async def restart(self):
        """Attempt to restart the connection after a failure"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            
        if self.task:
            try:
                self.task.cancel()
            except:
                pass
            
        # Wait a moment before reconnecting
        await asyncio.sleep(2)
        if self.active:  # Only restart if we're still supposed to be active
            await self.start()

    async def stop(self):
        """Stop the microphone handler gracefully"""
        print(f"[Mic {self.persona}] Stopping microphone handler...")
        self.active = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                print(f"[Mic {self.persona}] WebSocket closed")
            except Exception as e:
                print(f"[Mic {self.persona}] Error closing WebSocket: {str(e)}")
        
        if self.task:
            try:
                self.task.cancel()
                print(f"[Mic {self.persona}] Task cancelled")
            except Exception as e:
                print(f"[Mic {self.persona}] Error cancelling task: {str(e)}")

def start_background_loop(loop):
    """Start the background event loop"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

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

@app.route('/api/start_session', methods=['POST'])
def start_session():
    global config, mic_handlers
    
    if config['session_active']:
        return jsonify({"status": "error", "message": "Session already active"})

    # Clear translation history
    translation_history.clear()
    
    try:
        # Create event loop for the Deepgram handlers
        loop = asyncio.new_event_loop()
        
        # Create the microphone handlers
        mic_handlers['A'] = MicrophoneHandler(
            persona='A',
            target_language=config['personB_target_language'],
            voice=config['voiceB'],
            loop=loop
        )
        
        mic_handlers['B'] = MicrophoneHandler(
            persona='B',
            target_language=config['personA_target_language'],
            voice=config['voiceA'],
            loop=loop
        )
        
        # Start the handlers in their own threads
        threading.Thread(target=mic_handlers['A'].run_async, daemon=True).start()
        threading.Thread(target=mic_handlers['B'].run_async, daemon=True).start()
        
        config['session_active'] = True
        return jsonify({"status": "success", "message": "Translation session started"})

    except Exception as e:
        print(f"Error starting session: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    global config, mic_handlers, loop
    
    if not config['session_active']:
        return jsonify({"status": "error", "message": "No active session"})

    try:
        # Schedule the stop tasks in the background loop
        if mic_handlers['A']:
            asyncio.run_coroutine_threadsafe(mic_handlers['A'].stop(), loop)
        if mic_handlers['B']:
            asyncio.run_coroutine_threadsafe(mic_handlers['B'].stop(), loop)
        
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

@socketio.on('audio')
def handle_audio(data):
    """Handle audio data received from the client"""
    global mic_handlers
    
    if not config['session_active']:
        return
    
    try:
        # Get persona from request
        persona = request.args.get('persona', 'A')
        
        if mic_handlers[persona] and mic_handlers[persona].websocket:
            # Send the audio data to Deepgram
            asyncio.run_coroutine_threadsafe(
                mic_handlers[persona].websocket.send(data), 
                mic_handlers[persona].loop
            )
    except Exception as e:
        print(f"Error handling audio from client: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
