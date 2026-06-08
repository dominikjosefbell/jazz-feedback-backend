"""
jazzfb — Solo-Klavier Jazz-Improvisations-Analyse.

Modellunabhaengige, regelbasierte Engine: Note-Events rein -> strukturierte,
musikalisch fundierte Analyse raus. Die Transkription (Audio->Noten) liefert
ein externes Modell (Basic Pitch, Onsets-and-Frames, Piano-to-MIDI-API, MIDI).
"""

from .core import Note, BeatGrid, Changes, ChordSpan, from_midi, from_basic_pitch
from .theory import parse_chord, Chord
from .separation import separate, Separated, Cluster
from .report import (analyze, rule_based_summary, build_feedback_prompt,
                     get_llm_feedback)

__all__ = [
    "Note", "BeatGrid", "Changes", "ChordSpan", "from_midi", "from_basic_pitch",
    "parse_chord", "Chord", "separate", "Separated", "Cluster",
    "analyze", "rule_based_summary", "build_feedback_prompt", "get_llm_feedback",
]
