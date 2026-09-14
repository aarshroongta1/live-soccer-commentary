"""Voices. A protocol, a silent stand-in, and ElevenLabs when a key is set."""

from commentary.voice.elevenlabs import ElevenLabsSpeaker, VoiceUnavailable
from commentary.voice.playback import AudioSink, FFplaySink, NullSink, PcmSink, default_sink
from commentary.voice.say import SaySpeaker
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker, Speaker, Utterance

__all__ = [
    "WORDS_PER_SECOND",
    "AudioSink",
    "ElevenLabsSpeaker",
    "FFplaySink",
    "LogSpeaker",
    "NullSink",
    "PcmSink",
    "SaySpeaker",
    "Speaker",
    "Utterance",
    "VoiceUnavailable",
    "default_sink",
]
