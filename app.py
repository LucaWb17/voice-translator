# app.py
from flask import Flask, render_template, request, jsonify, Response
import json
import os
import threading
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from elevenlabs import set_api_key
from elevenlabs import play, generate
import assemblyai as aai
from langdetect import detect
import base64

app = Flask(__name__)
load_dotenv()

# Global variables to store configuration
config = {
    "personA_target_language": "English",
    "personB_target_language": "German",
    "voiceA": "George",
    "voiceB": "Arnold",
    "session_active": False
}

# Dictionary to store available voices
available_voices = {
    "English": ["George", "Rachel", "Emily"],
    "Italian": ["Antonio", "Isabella"],
    "German": ["Arnold", "Klaus"],
    "French": ["Pierre", "Sophie"],
    "Spanish": ["Carlos", "Elena"],
    "Japanese": ["Hiroshi", "Yuki"]
}

# Available languages
available_languages = [
    "English", "Italian", "German", "French", 
    "Spanish", "Japanese", "Portuguese", "Russian", 
    "Chinese", "Arabic", "Hindi", "Dutch", "Czech"
]

# === GPT setup ===
translation_template = """
Translate the following sentence into {language}, return ONLY the translation, nothing else.

Sentence: {sentence}
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

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
    audio = client.generate(
        text=text,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    return audio

# AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Global list to store translation history
translation_history = []

# Microphone handlers
mic_handlers = {'A': None, 'B': None}

class MicrophoneHandler:
    def __init__(self, persona, target_language, voice):
        self.persona = persona
        self.target_language = target_language
        self.voice = voice
        self.active = False
        self.transcriber = None
        self.thread = None
    
    def start(self, device_index=None):
        self.active = True
        
        def on_open(session_opened: aai.RealtimeSessionOpened):
            print(f"[Mic {self.persona}] ✅ Session started - ID: {session_opened.session_id}")
        
        def on_data(transcript: aai.RealtimeTranscript):
            if not transcript.text or not self.active:
                return
            
            if isinstance(transcript, aai.RealtimeFinalTranscript):
                try:
                    detected_lang = detect(transcript.text)
                except:
                    detected_lang = "unknown"
                
                translation = translate(transcript.text, self.target_language)
                audio = gen_dub(translation, voice=self.voice)
                audio_b64 = base64.b64encode(audio).decode('utf-8')
                
                translation_entry = {
                    "timestamp": time.time(),
                    "persona": self.persona,
                    "original_text": transcript.text,
                    "detected_language": detected_lang,
                    "translated_text": translation,
                    "target_language": self.target_language,
                    "voice": self.voice,
                    "audio_b64": audio_b64
                }
                
                translation_history.append(translation_entry)
                # Keep only last 50 translations
                if len(translation_history) > 50:
                    translation_history.pop(0)
        
        def on_error(error: aai.RealtimeError):
            print(f"[Mic {self.persona}] ❌ ERROR: {error}")
        
        def on_close():
            print(f"[Mic {self.persona}] 🔒 Session closed.")
        
        def run_transcriber():
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=44_100,
                on_data=on_data,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close,
                device_index=device_index
            )
            
            self.transcriber.connect()
            mic_stream = aai.extras.MicrophoneStream(device_index=device_index)
            self.transcriber.stream(mic_stream)
            
            while self.active:
                time.sleep(0.1)
            
            self.transcriber.close()
        
        self.thread = threading.Thread(target=run_transcriber)
        self.thread.start()
    
    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join(timeout=2)
        self.thread = None

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
    
    # Clear previous history
    translation_history.clear()
    
    try:
        # Create and start microphone handlers
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
        
        # Start with device indices
        mic_handlers['A'].start(device_index=0)
        mic_handlers['B'].start(device_index=1)
        
        config['session_active'] = True
        return jsonify({"status": "success", "message": "Translation session started"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_session', methods=['POST'])
def stop_session():
    global config, mic_handlers
    
    if not config['session_active']:
        return jsonify({"status": "error", "message": "No active session"})
    
    try:
        # Stop microphone handlers
        if mic_handlers['A']:
            mic_handlers['A'].stop()
        if mic_handlers['B']:
            mic_handlers['B'].stop()
        
        mic_handlers = {'A': None, 'B': None}
        config['session_active'] = False
        
        return jsonify({"status": "success", "message": "Translation session stopped"})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/translation_history')
def get_history():
    # Option to get only entries after a certain timestamp
    since = request.args.get('since', 0, type=float)
    filtered_history = [entry for entry in translation_history if entry['timestamp'] > since]
    return jsonify(filtered_history)

@app.route('/api/available_languages')
def get_languages():
    return jsonify(available_languages)

@app.route('/api/available_voices')
def get_voices():
    return jsonify(available_voices)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
