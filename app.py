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
from elevenlabs import set_api_key, generate, play
from deepgram import Deepgram
from langdetect import detect
import base64
import asyncio
import pyaudio
import websockets

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

# Deepgram setup
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram = Deepgram(DEEPGRAM_API_KEY)

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
        self.websocket = None
        self.thread = None
    
    async def start(self, device_index=None):
        self.active = True
        
        try:
            # Inizializza la connessione WebSocket con Deepgram
            self.websocket = await deepgram.transcription.live({
                'encoding': 'linear16',
                'sample_rate': 44100,
                'channels': 1,
                'language': 'auto',
                'model': 'nova-2',
                'smart_format': True,
                'interim_results': True
            })

             # Funzione per catturare l'audio e inviarlo al WebSocket
        async def stream_audio():
            p = pyaudio.PyAudio()
            # Apre il flusso del microfono
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024,
                input_device_index=device_index  # se hai un dispositivo specifico, altrimenti lascialo vuoto
            )
            
            while self.active:
                data = stream.read(1024, exception_on_overflow=False)  # Leggi l'audio dal microfono
                await self.websocket.send(data)  # Invia l'audio al WebSocket

        # Funzione per gestire la trascrizione dal WebSocket
        async def handle_transcription():
            while self.active:
                try:
                    message = await self.websocket.receive()  # Aspetta la risposta dal WebSocket
                    data = json.loads(message)
                    
                    if 'channel' in data and 'alternatives' in data['channel']:
                        transcript = data['channel']['alternatives'][0]['transcript']  # Prendi la trascrizione

                        if transcript:
                            # Traduci il testo ricevuto
                            detected_lang = detect(transcript)
                            translation = translate(transcript, self.target_language)
                            audio = gen_dub(translation, voice=self.voice)
                            audio_b64 = base64.b64encode(audio).decode('utf-8')

                            # Salva la traduzione nella cronologia
                            translation_entry = {
                                "timestamp": time.time(),
                                "persona": self.persona,
                                "original_text": transcript,
                                "detected_language": detected_lang,
                                "translated_text": translation,
                                "target_language": self.target_language,
                                "voice": self.voice,
                                "audio_b64": audio_b64
                            }
                            translation_history.append(translation_entry)
                            if len(translation_history) > 50:
                                translation_history.pop(0)

                except Exception as e:
                    print(f"[Mic {self.persona}] ❌ ERROR: {str(e)}")

        # Avvia entrambe le coroutine in parallelo
        self.thread = asyncio.gather(stream_audio(), handle_transcription())
    
    except Exception as e:
        print(f"[Mic {self.persona}] ❌ ERROR: {str(e)}")
        self.active = False
            
            async def handle_transcription():
                while self.active:
                    try:
                        message = await self.websocket.receive()
                        data = json.loads(message)
                        
                        if 'channel' in data and 'alternatives' in data['channel']:
                            transcript = data['channel']['alternatives'][0]['transcript']
                            
                            if transcript and self.active:
                                try:
                                    detected_lang = detect(transcript)
                                except:
                                    detected_lang = "unknown"
                                
                                translation = translate(transcript, self.target_language)
                                audio = gen_dub(translation, voice=self.voice)
                                audio_b64 = base64.b64encode(audio).decode('utf-8')
                                
                                translation_entry = {
                                    "timestamp": time.time(),
                                    "persona": self.persona,
                                    "original_text": transcript,
                                    "detected_language": detected_lang,
                                    "translated_text": translation,
                                    "target_language": self.target_language,
                                    "voice": self.voice,
                                    "audio_b64": audio_b64
                                }
                                
                                translation_history.append(translation_entry)
                                if len(translation_history) > 50:
                                    translation_history.pop(0)
                    
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"[Mic {self.persona}] ❌ ERROR: {str(e)}")
            
            # Avvia il loop di gestione della trascrizione
            self.thread = asyncio.create_task(handle_transcription())
            
        except Exception as e:
            print(f"[Mic {self.persona}] ❌ ERROR: {str(e)}")
            self.active = False
    
    async def stop(self):
        self.active = False
        if self.websocket:
            await self.websocket.close()
        if self.thread:
            await self.thread

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
    
    translation_history.clear()
    
    try:
        # Creazione dei gestori del microfono
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
        
        # Avvio delle sessioni in modo asincrono
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(mic_handlers['A'].start(device_index=0))
        loop.run_until_complete(mic_handlers['B'].start(device_index=1))
        
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
        # Ferma i gestori del microfono in modo asincrono
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if mic_handlers['A']:
            loop.run_until_complete(mic_handlers['A'].stop())
        if mic_handlers['B']:
            loop.run_until_complete(mic_handlers['B'].stop())
        
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
