"""
core.py — Noten-Events, Loader (MIDI optional), Beat-Raster und Changes/Form.

Die Engine arbeitet auf Note-Events; woher sie kommen (Transkriptionsmodell,
Piano-to-MIDI-API, MIDI-Datei) ist egal.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .theory import Chord, parse_chord, pitch_to_pc


# --- Note -------------------------------------------------------------------

@dataclass
class Note:
    onset: float       # Sekunden
    offset: float      # Sekunden
    pitch: int         # MIDI-Tonhoehe (60 = C4)
    velocity: int = 80

    @property
    def duration(self) -> float:
        return max(0.0, self.offset - self.onset)

    @property
    def pc(self) -> int:
        return pitch_to_pc(self.pitch)


def from_midi(path: str) -> list[Note]:
    """Laedt Noten aus einer MIDI-Datei. Versucht pretty_midi, dann mido."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(path)
        notes = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                notes.append(Note(n.start, n.end, n.pitch, n.velocity))
        return sorted(notes, key=lambda x: (x.onset, x.pitch))
    except ImportError:
        pass
    try:
        import mido
    except ImportError as e:
        raise ImportError(
            "Fuer MIDI-Laden bitte 'pip install pretty_midi' (empfohlen) "
            "oder 'pip install mido'."
        ) from e
    mid = mido.MidiFile(path)
    notes, on = [], {}
    t = 0.0
    for msg in mido.merge_tracks(mid.tracks):
        t += mido.tick2second(msg.time, mid.ticks_per_beat, 500000)
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = (t, msg.velocity)
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in on:
                st, vel = on.pop(msg.note)
                notes.append(Note(st, t, msg.note, vel))
    return sorted(notes, key=lambda x: (x.onset, x.pitch))


def from_basic_pitch(note_events) -> list[Note]:
    """Adapter fuer Spotify Basic Pitch: dessen note_events sind Tupel
    (start_sec, end_sec, pitch_midi, amplitude, [pitch_bends])."""
    out = []
    for ev in note_events:
        start, end, pitch = ev[0], ev[1], int(ev[2])
        amp = ev[3] if len(ev) > 3 else 0.8
        out.append(Note(start, end, pitch, int(max(1, min(127, amp * 127)))))
    return sorted(out, key=lambda x: (x.onset, x.pitch))


# --- Beat-Raster ------------------------------------------------------------

@dataclass
class BeatGrid:
    """Einfaches Raster bei festem Tempo. Fuer variables Tempo kann man
    extern (madmom/librosa) Beats tracken und 'beat_times' direkt setzen."""
    bpm: float
    start: float = 0.0          # Zeit (s) des ersten Downbeats
    beats_per_bar: int = 4
    beat_times: Optional[list[float]] = None   # optional: getrackte Beats

    @property
    def spb(self) -> float:                     # Sekunden pro Beat
        return 60.0 / self.bpm

    def position(self, t: float) -> tuple[int, float]:
        """Gibt (Takt ab 0, Beat-Phase 0..beats_per_bar) fuer Zeit t."""
        beats = (t - self.start) / self.spb
        bar = int(beats // self.beats_per_bar)
        beat_in_bar = beats - bar * self.beats_per_bar
        return bar, beat_in_bar

    def is_strong(self, t: float) -> bool:
        """Betonte Zeit = nahe an Beat 1 oder 3 (im 4/4)."""
        _, b = self.position(t)
        nearest = round(b)
        if abs(b - nearest) > 0.18:        # deutlich neben dem Beat -> schwach
            return False
        return nearest % 2 == 0            # Beats 1 und 3

    def beat_phase(self, t: float) -> float:
        """Signierte Abweichung vom naechsten Beat in Beat-Bruchteilen
        (negativ = vor dem Beat / 'laid back' positiv = hinter dem Beat)."""
        beats = (t - self.start) / self.spb
        return beats - round(beats)


# --- Changes / Form ---------------------------------------------------------

@dataclass
class ChordSpan:
    bar: int                 # Takt ab 0
    beat: float              # Start-Beat im Takt (0-basiert)
    beats: float             # Dauer in Beats
    symbol: str
    chord: Chord = field(init=False)

    def __post_init__(self):
        self.chord = parse_chord(self.symbol)


@dataclass
class Changes:
    spans: list[ChordSpan]
    beats_per_bar: int = 4

    @classmethod
    def from_bars(cls, bars: list[list[str]], beats_per_bar: int = 4) -> "Changes":
        """Komfort-Konstruktor: pro Takt 1 oder 2 (oder n) Akkorde, die den
        Takt gleichmaessig teilen. bars = [["Dm7"], ["G7"], ["Cmaj7","A7"], ...]"""
        spans = []
        for i, bar in enumerate(bars):
            n = len(bar)
            dur = beats_per_bar / n
            for j, sym in enumerate(bar):
                spans.append(ChordSpan(i, j * dur, dur, sym))
        return cls(spans, beats_per_bar)

    def chord_at(self, bar: int, beat: float) -> Optional[ChordSpan]:
        form_len = max((s.bar for s in self.spans), default=0) + 1
        bar = bar % form_len                     # Form wiederholt sich
        best = None
        for s in self.spans:
            if s.bar == bar and s.beat <= beat + 1e-6 < s.beat + s.beats:
                return s
            if s.bar == bar and s.beat <= beat + 1e-6:
                best = s
        return best
