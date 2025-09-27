import os
import time
import wave
import tempfile
import logging
import asyncio
import numpy as np
import sounddevice as sd
import simpleaudio as sa
import requests
from dotenv import load_dotenv
from agents import Runner
from .agent_builder import build_main_agent
from .state import get_state

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in .env")

PERSONA = (os.getenv("PERSONA") or "business").lower().strip()

API_BASE = "https://api.openai.com/v1"
STT_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
VOICE = "alloy"
AUDIO_FORMAT = "mp3"

SAMPLE_RATE = 16000
CHANNELS = 1
DT_SECONDS = 4

# Initialize AI agent
app_state = get_state(user_name="Николай", persona=PERSONA)
agent = build_main_agent(app_state)


def test_microphone(device_id: int = None) -> bool:
    """Test microphone functionality for specific device."""
    print("🎤 Checking available microphones:")
    
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    for i, device in enumerate(input_devices):
        device_idx = devices.index(device)
        status = "← SELECTED" if device_idx == device_id else ""
        print(f"  {i}: {device['name']} (channels: {device['max_input_channels']}) {status}")
    
    # Test specific device
    try:
        device_name = "default"
        if device_id is not None:
            device_name = devices[device_id]['name']
            
        print(f"\n🔊 Testing microphone: {device_name} (2 seconds, say something)...")
        
        test_audio = sd.rec(
            int(2 * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype="int16",
            device=device_id
        )
        sd.wait()
        
        level = np.max(np.abs(test_audio))
        mean_level = np.mean(np.abs(test_audio))
        
        print(f"📊 Maximum level: {level}")
        print(f"📊 Average level: {mean_level:.1f}")
        
        if level < 100:
            print("⚠️  Microphone is too quiet or not working!")
            print("   Try:")
            print("   1. Check system sound settings")
            print("   2. Increase microphone volume") 
            print("   3. Select a different microphone")
            print("   4. Speak louder during the test")
            return False
        elif level < 1000:
            print("⚠️  Microphone works, but signal is weak")
            print("   Recommended to increase volume")
            return True
        else:
            print("✅ Microphone works perfectly!")
            return True
            
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        print(f"❌ Microphone test error: {e}")
        return False


def select_microphone() -> int:
    """Allow user to select a microphone device."""
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    print("\nSelect microphone:")
    for i, device in enumerate(input_devices):
        print(f"  {i}: {device['name']}")
    print(f"  {len(input_devices)}: Use default")
    
    while True:
        try:
            choice = input("Введите номер (Enter для умолчания): ").strip()
            if not choice or choice == str(len(input_devices)):
                device_id = None
                break
            
            device_idx = int(choice)
            if 0 <= device_idx < len(input_devices):
                selected_device = input_devices[device_idx]
                device_id = devices.index(selected_device)
                print(f"✅ Selected: {selected_device['name']}")
                break
            else:
                print(f"⚠️  Invalid number, try again")
                continue
                
        except (ValueError, IndexError):
            print(f"⚠️  Invalid input, try again")
            continue
    
    # Test selected microphone
    print(f"\n{'='*50}")
    if test_microphone(device_id):
        return device_id
    else:
        retry = input("\nПопробовать другой микрофон? (y/N): ").lower().strip()
        if retry in ['y', 'yes', 'да']:
            return select_microphone()  # Recursive retry
        else:
            print(f"Continuing with current settings...")
            return device_id


def record_wav(
    path: str, 
    seconds: int = DT_SECONDS,
    device_id: int = None
) -> None:
    """Record microphone audio and save as 16-bit PCM WAV."""
    logger.info(f"Starting recording for {seconds} seconds...")
    
    frames = int(seconds * SAMPLE_RATE)
    
    try:
        audio = sd.rec(
            frames, 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS, 
            dtype="int16",
            device=device_id
        )
        sd.wait()
        
        # Check if recording has actual audio data
        audio_level = np.max(np.abs(audio))
        logger.info(f"Recording completed. Max audio level: {audio_level}")
        
        if audio_level < 100:
            logger.warning("Recording seems too quiet. Check microphone settings.")
            
        # Write WAV header + PCM data
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
            
        file_size = os.path.getsize(path)
        logger.info(f"WAV file saved: {path} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        raise


def transcribe_wav(path: str) -> str:
    """Send WAV to OpenAI STT and return transcribed text."""
    logger.info(f"Transcribing audio file: {path}")
    
    # Check if file exists and has content
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    file_size = os.path.getsize(path)
    if file_size < 1000:  # Less than 1KB suggests empty recording
        logger.warning(f"Audio file seems too small: {file_size} bytes")
    
    url = f"{API_BASE}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    try:
        with open(path, "rb") as audio_file:
            files = {
                "file": ("speech.wav", audio_file, "audio/wav"),
            }
            data = {
                "model": STT_MODEL,
                "language": "ru",  # Explicitly set Russian language
                "response_format": "json"
            }
            
            logger.info("Sending audio to OpenAI Whisper API...")
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            resp.raise_for_status()
            
            payload = resp.json()
            transcribed_text = payload.get("text", "").strip()
            
            logger.info(f"Transcription result: '{transcribed_text}'")
            logger.info(f"Detected language: {payload.get('language', 'unknown')}")
            
            return transcribed_text
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return f"Recognition error: {e}"


async def process_text_with_agent(
    user_text: str,
    conversation_history: list
) -> str:
    """Process user text using the AI agent."""
    if not user_text:
        return "Я ничего не расслышал."
    
    try:
        conversation_history.append({"role": "user", "content": user_text})
        
        result = await Runner.run(agent, conversation_history)
        
        assistant_reply = result.final_output if hasattr(result, "final_output") else str(result)
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply
    except Exception as e:
        logger.error(f"Agent processing failed: {e}")
        return f"Sorry, an error occurred while processing the request: {e}"


def tts_to_wav(
    text: str, 
    out_path: str
) -> None:
    """Synthesize speech from text and save to audio file."""
    url = f"{API_BASE}/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": TTS_MODEL,
        "voice": VOICE,
        "input": text,
        "response_format": AUDIO_FORMAT,
    }
    
    try:
        resp = requests.post(url, headers=headers, json=json_data, timeout=120)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        logger.debug(f"TTS audio saved to {out_path}")
    except Exception as e:
        logger.error(f"TTS request failed: {e}")
        raise


def play_audio(path: str) -> None:
    """Play audio file. Handle MP3 format by converting to WAV first."""
    try:
        # For MP3 files, we need to use a different approach
        if path.endswith('.mp3'):
            # Use system command to play MP3
            import subprocess
            subprocess.run(['mpg123', '-q', path], check=True)
        else:
            # For WAV files, use simpleaudio
            wave_obj = sa.WaveObject.from_wave_file(path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
    except FileNotFoundError:
        logger.error("mpg123 not found. Install it with: sudo apt-get install mpg123")
        raise
    except Exception as e:
        logger.error(f"Audio playback failed: {e}")
        raise


async def main() -> None:
    print(f"=== Voice Assistant ===")
    
    # Let user select and test microphone
    device_id = select_microphone()
    
    print(f"\nActive persona: {PERSONA}")
    print(f"Говорите 'stop' для exitа\n")
    
    conversation_history = []
    
    with tempfile.TemporaryDirectory() as tmp:
        while True:
            wav_in = os.path.join(tmp, f"in_{int(time.time())}.wav")
            audio_out = os.path.join(tmp, f"out_{int(time.time())}.{AUDIO_FORMAT}")

            print(f"\n📼 Recording... (speak now)")
            record_wav(wav_in, DT_SECONDS, device_id)

            print(f"🎯 Recognizing speech...")
            user_text = transcribe_wav(wav_in)
            print(f"👤 You said: '{user_text}'")

            if user_text.lower().strip() in ["stop", "stop", "exit", "exit"]:
                print(f"👋 Goodbye!")
                break
                
            if (not user_text or 
                user_text.lower().strip() in ["you"] or
                "редактор субтитров" in user_text.lower() or
                len(user_text.strip()) < 2):
                print(f"⚠️  Failed to recognize speech. Try speaking louder and clearer.")
                continue

            print(f"🤖 Processing with AI agent...")
            result_text = await process_text_with_agent(user_text, conversation_history)
            print(f"🤖 Bot: {result_text}")

            print(f"🔊 Generating speech...")
            tts_to_wav(result_text, audio_out)
            play_audio(audio_out)


if __name__ == "__main__":
    asyncio.run(main())
