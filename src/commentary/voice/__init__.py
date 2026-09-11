"""Voices. A protocol, a silent stand-in, and ElevenLabs when a key is set."""

from commentary.voice.elevenlabs import ElevenLabsSpeaker, speaker_from_env
from commentary.voice.playback import AudioSink, FFplaySink, NullSink, default_sink
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker, Speaker, Utterance

__all__ = [
    "WORDS_PER_SECOND",
    "AudioSink",
    "ElevenLabsSpeaker",
    "FFplaySink",
    "LogSpeaker",
    "NullSink",
    "Speaker",
    "Utterance",
    "default_sink",
    "speaker_from_env",
]
