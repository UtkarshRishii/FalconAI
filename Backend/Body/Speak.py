import pygame
import random
import asyncio
import edge_tts
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv

load_dotenv()

class TextToSpeechEngine:
    """A clean and efficient Text-to-Speech engine using edge-tts and pygame."""
    
    def __init__(self, audio_dir: str = "Database/Audio"):
        """
        Initialize the TTS engine.
        
        Args:
            audio_dir: Directory to store audio files
            voice: Voice to use (defaults to environment variable)
        """
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_file = self.audio_dir / "speak.mp3"
        
        self.voice = os.getenv('Voice')
        self.pitch = '+0Hz'
        self.rate = '+0%'
        
        # Truncation responses for long text
        self.truncation_responses = [
            "The rest of the result has been printed to the chat screen.",
            "You can find the complete text on the chat screen.",
            "Please check the chat screen for the full response.",
            "The remaining content is displayed on your screen.",
            "The complete answer is available on the chat screen."
        ]
        
        # Text processing settings
        self.max_sentences_for_long_text = 2
        self.min_length_for_truncation = 250
        self.max_sentences_threshold = 8

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing emojis and special characters.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text suitable for TTS
        """
        # Remove emojis and non-printable characters
        cleaned = ''.join(
            char for char in text 
            if char.isprintable() and not unicodedata.category(char).startswith('So')
        )
        
        # Keep only alphanumeric, spaces, and basic punctuation
        cleaned = re.sub(r"[^\w\s.,!?'\-]", "", cleaned)
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    async def _generate_audio(self, text: str) -> None:
        """Generate audio file from text using edge-tts."""
        # Remove existing file if it exists
        if self.audio_file.exists():
            self.audio_file.unlink()
        
        communicate = edge_tts.Communicate(
            text, 
            self.voice, 
            pitch=self.pitch, 
            rate=self.rate
        )
        await communicate.save(str(self.audio_file))

    def _play_audio(self, stop_callback: Callable[[], bool]) -> None:
        """Play the generated audio file."""
        try:
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Load and play audio
            pygame.mixer.music.load(str(self.audio_file))
            pygame.mixer.music.play()
            
            # Wait for playback to complete or stop callback
            clock = pygame.time.Clock()
            while pygame.mixer.music.get_busy():
                if not stop_callback():
                    pygame.mixer.music.stop()
                    break
                clock.tick(10)
                
        except pygame.error as e:
            print(f"Pygame error: {e}")
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            self._cleanup_audio()

    def _cleanup_audio(self) -> None:
        """Clean up pygame resources."""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def _should_truncate_text(self, text: str) -> bool:
        """Determine if text should be truncated for speech."""
        sentences = text.split('.')
        return (len(sentences) > self.max_sentences_threshold and 
                len(text) >= self.min_length_for_truncation)

    def _prepare_speech_text(self, text: str) -> str:
        """Prepare text for speech, truncating if necessary."""
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text.strip():
            return "No text to speak."
        
        if self._should_truncate_text(cleaned_text):
            sentences = cleaned_text.split('.')
            truncated = '. '.join(sentences[:self.max_sentences_for_long_text])
            if truncated and not truncated.endswith('.'):
                truncated += '.'
            
            truncation_msg = random.choice(self.truncation_responses)
            return f"{truncated} {truncation_msg}"
        
        return cleaned_text

    async def speak_async(self, text: str, stop_callback: Callable[[], bool] = lambda: True) -> None:
        """
        Convert text to speech asynchronously.
        
        Args:
            text: Text to convert to speech
            stop_callback: Function that returns False to stop playback
        """
        if not text or not text.strip():
            return
        
        speech_text = self._prepare_speech_text(text)
        
        try:
            await self._generate_audio(speech_text)
            self._play_audio(stop_callback)
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            stop_callback()

    def speak(self, text: str, stop_callback: Callable[[], bool] = lambda: True) -> None:
        """
        Convert text to speech (synchronous wrapper).
        
        Args:
            text: Text to convert to speech
            stop_callback: Function that returns False to stop playback
        """
        try:
            asyncio.run(self.speak_async(text, stop_callback))
        except Exception as e:
            print(f"TTS execution error: {e}")

    def set_voice_settings(self, voice: str = None, pitch: str = None, rate: str = None) -> None:
        """
        Update voice settings.
        
        Args:
            voice: Voice identifier
            pitch: Pitch adjustment (e.g., '+0Hz', '+50Hz')
            rate: Speech rate (e.g., '+0%', '+50%')
        """
        if voice:
            self.voice = voice
        if pitch:
            self.pitch = pitch
        if rate:
            self.rate = rate


# Convenience functions for backward compatibility
def clean_text(text: str) -> str:
    """Clean text for TTS processing."""
    return TextToSpeechEngine.clean_text(text)

def Speak(text: str, func: Callable[[], bool] = lambda: True) -> None:
    """Simple TTS function for backward compatibility."""
    engine = TextToSpeechEngine()
    engine.speak(text, func)