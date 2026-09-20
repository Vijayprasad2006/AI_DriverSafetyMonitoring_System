"""
DriveGuard AI - Audio Alarm System
Multi-backend non-blocking sound alerts for WARNING and CRITICAL fatigue levels.
Utilizes threaded winsound on Windows with pygame/sounddevice fallback and cooldown logic.
"""

import time
import threading
import math
import wave
import struct
from pathlib import Path
from typing import Optional
import config

class AudioAlertSystem:
    """Provides non-blocking audio alerts with cooldowns and volume control."""

    def __init__(self, volume: float = config.DEFAULT_AUDIO_VOLUME):
        self.volume = max(0.0, min(1.0, volume))
        self.last_warning_time = 0.0
        self.last_critical_time = 0.0
        self.warning_cooldown = config.AUDIO_COOLDOWN_WARNING
        self.critical_cooldown = config.AUDIO_COOLDOWN_CRITICAL
        self.is_playing = False
        self.enabled = True
        
        # Pre-generate wav sound files if not existing
        self._ensure_sound_assets()

    def _ensure_sound_assets(self):
        """Generates warning and critical WAV files if they do not exist."""
        warning_wav = config.SOUNDS_DIR / "warning.wav"
        critical_wav = config.SOUNDS_DIR / "critical.wav"

        if not warning_wav.exists():
            self._create_tone_wav(str(warning_wav), [(600, 0.15), (0, 0.05), (600, 0.15)])
        if not critical_wav.exists():
            self._create_tone_wav(str(critical_wav), [(1000, 0.12), (1400, 0.12), (1000, 0.12), (1400, 0.18)])

    def _create_tone_wav(self, file_path: str, tones: list, sample_rate: int = 22050):
        """Synthesizes a multi-frequency wave file for alerts."""
        try:
            with wave.open(file_path, "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                
                for freq, duration in tones:
                    num_samples = int(sample_rate * duration)
                    for i in range(num_samples):
                        if freq == 0:
                            sample = 0
                        else:
                            t = float(i) / sample_rate
                            sample = int(32767.0 * 0.7 * math.sin(2.0 * math.pi * freq * t))
                        data = struct.pack("<h", sample)
                        wav_file.writeframesraw(data)
        except Exception as e:
            pass

    def play_warning(self, force: bool = False):
        """Plays short warning pulse if cooldown has passed."""
        if not self.enabled or self.volume <= 0.0:
            return

        now = time.time()
        if force or (now - self.last_warning_time >= self.warning_cooldown):
            self.last_warning_time = now
            threading.Thread(target=self._play_warning_async, daemon=True).start()

    def play_critical(self, force: bool = False):
        """Plays urgent emergency siren if cooldown has passed."""
        if not self.enabled or self.volume <= 0.0:
            return

        now = time.time()
        if force or (now - self.last_critical_time >= self.critical_cooldown):
            self.last_critical_time = now
            threading.Thread(target=self._play_critical_async, daemon=True).start()

    def _play_warning_async(self):
        """Background thread executing warning sound."""
        try:
            import winsound
            winsound.Beep(750, 200)
            time.sleep(0.08)
            winsound.Beep(750, 250)
            return
        except Exception:
            pass

        # Fallback to pygame or wav
        self._play_wav(config.SOUNDS_DIR / "warning.wav")

    def _play_critical_async(self):
        """Background thread executing critical alarm sequence."""
        try:
            import winsound
            for _ in range(3):
                winsound.Beep(1200, 150)
                time.sleep(0.05)
                winsound.Beep(1600, 200)
                time.sleep(0.05)
            return
        except Exception:
            pass

        # Fallback to wav
        self._play_wav(config.SOUNDS_DIR / "critical.wav")

    def _play_wav(self, path: Path):
        """Plays a WAV file via pygame if available."""
        if not path.exists():
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            sound = pygame.mixer.Sound(str(path))
            sound.set_volume(self.volume)
            sound.play()
        except Exception:
            pass
