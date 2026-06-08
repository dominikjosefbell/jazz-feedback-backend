"""
theory.py — Musiktheorie-Kern: Tonhöhenklassen, Akkordsymbol-Parser,
Akkord-Skalen-Tabellen und Klassifikation einzelner Töne gegen einen Akkord.

Alles in Halbtonschritten (Intervall = (pitchclass - root) % 12) gerechnet.
Bewusst regelbasiert: kein Training, kein Korpus, kein Lizenzproblem.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re

# --- Tonhöhenklassen --------------------------------------------------------

NOTE_TO_PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11, "B#": 0,
}
PC_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pitch_to_pc(midi_pitch: int) -> int:
    return midi_pitch % 12


def pc_name(pc: int) -> str:
    return PC_TO_NAME[pc % 12]


# --- Akkord-Modell ----------------------------------------------------------
# Pro Basis-Qualitaet: chord_tones (Akkordtoene), tensions (verfuegbare
# Optionstoene), avoid (zu vermeiden auf betonten Zeiten). Intervalle in
# Halbtonschritten ueber dem Grundton. Vereinfachte, gaengige Jazz-Konvention.

BASE_QUALITIES = {
    "maj7":   dict(tones=[0, 4, 7, 11], tensions=[2, 9 % 12, 6],  avoid=[5]),   # Ionisch/Lydisch (9, 13, #11), Avoid 11
    "maj6":   dict(tones=[0, 4, 7, 9],  tensions=[2, 6],          avoid=[5]),
    "dom7":   dict(tones=[0, 4, 7, 10], tensions=[2, 9],          avoid=[5]),   # Mixolydisch (9, 13), Avoid 11
    "min7":   dict(tones=[0, 3, 7, 10], tensions=[2, 5, 9],       avoid=[]),    # Dorisch (9, 11, 13)
    "min6":   dict(tones=[0, 3, 7, 9],  tensions=[2, 5],          avoid=[]),    # Melodisch-Moll/Dorisch
    "minMaj7":dict(tones=[0, 3, 7, 11], tensions=[2, 5, 9],       avoid=[]),
    "min7b5": dict(tones=[0, 3, 6, 10], tensions=[5, 8, 2],       avoid=[1]),   # Lokrisch #2 (11, b13, 9), Avoid b9
    "dim7":   dict(tones=[0, 3, 6, 9],  tensions=[2, 5, 8, 11],   avoid=[]),    # Ganzton-Halbton
    "aug7":   dict(tones=[0, 4, 8, 10], tensions=[2, 6],          avoid=[]),    # 7#5
    "sus7":   dict(tones=[0, 5, 7, 10], tensions=[2, 9],          avoid=[]),    # 7sus4
}

# Alterationen fuer Dominanten: Token -> hinzuzufuegende Tension-Intervalle
ALTERATIONS = {
    "b9": [1], "#9": [3], "#11": [6], "b5": [6], "b13": [8], "#5": [8],
}


@dataclass
class Chord:
    symbol: str
    root: int                 # Tonhoehenklasse 0-11
    quality: str              # Basis-Qualitaetsschluessel
    tones: list[int]          # absolute Tonhoehenklassen
    tensions: list[int]
    avoid: list[int]

    def classify(self, pc: int) -> str:
        """Klassifiziert eine Tonhoehenklasse relativ zum Akkord."""
        if pc in self.tones:
            return "chord_tone"
        if pc in self.tensions:
            return "tension"
        if pc in self.avoid:
            return "avoid"
        return "chromatic"

    def guide_tones(self) -> list[int]:
        """3 und 7 (bzw. b3/b7) als absolute Tonhoehenklassen — die Leittoene."""
        third = next((t for t in self.tones if (t - self.root) % 12 in (3, 4)), None)
        sev = next((t for t in self.tones if (t - self.root) % 12 in (10, 11)), None)
        return [x for x in (third, sev) if x is not None]


_ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)")


def parse_chord(symbol: str) -> Chord:
    """Parst gaengige Jazz-Akkordsymbole, z.B. Cmaj7, Dm7, G7alt, Bb7#11,
    F#m7b5, Ebmaj7#11, A7b9, Cm6, C7sus4, Co7."""
    s = symbol.strip()
    m = _ROOT_RE.match(s)
    if not m:
        raise ValueError(f"Akkord nicht lesbar: {symbol!r}")
    letter = m.group(1).upper()
    accidental = m.group(2)
    root = NOTE_TO_PC[(letter + accidental).upper()]
    rest = s[m.end():]
    rl = rest.lower()

    # Basis-Qualitaet bestimmen
    if rl.startswith(("maj7", "ma7", "M7", "Δ")) or rl in ("maj7",):
        base = "maj7"
    elif "maj7" in rl or "ma7" in rl or "Δ" in rest:
        base = "maj7"
    elif rl.startswith("6") and ("m" not in rl[:1]):
        base = "maj6"
    elif rl.startswith(("m6", "min6", "-6")):
        base = "min6"
    elif rl.startswith(("mmaj7", "m(maj7)", "minmaj7", "-maj7")):
        base = "minMaj7"
    elif rl.startswith(("m7b5", "min7b5", "-7b5")) or "ø" in rest or "m7-5" in rl:
        base = "min7b5"
    elif rl.startswith(("dim7", "o7", "°7")) or rl in ("dim", "o", "°"):
        base = "dim7"
    elif rl.startswith(("m7", "min7", "-7", "m9", "m11", "m13")) or rl == "m":
        base = "min7"
    elif "sus" in rl:
        base = "sus7"
    elif rl.startswith(("7#5", "7+", "aug7", "+7")):
        base = "aug7"
    elif rl.startswith("7") or "7" in rl or rl == "" and False:
        base = "dom7"
    elif rl == "":
        base = "maj7"   # blanker Grossbuchstabe -> Dur-Tonika als maj7 behandelt
    else:
        base = "dom7"

    spec = BASE_QUALITIES[base]
    tones = sorted({(root + i) % 12 for i in spec["tones"]})
    tensions = {(root + i) % 12 for i in spec["tensions"]}
    avoid = {(root + i) % 12 for i in spec["avoid"]}

    # Alterationen anwenden (vor allem fuer Dominanten)
    if base in ("dom7", "aug7"):
        if "alt" in rl:
            for toks in ("b9", "#9", "#11", "b13"):
                tensions |= {(root + ALTERATIONS[toks][0]) % 12}
            tensions.discard((root + 2) % 12)   # natuerliche 9 raus
            tensions.discard((root + 9) % 12)   # natuerliche 13 raus
            avoid = set()
        for tok, semis in ALTERATIONS.items():
            if tok in rl:
                tensions |= {(root + x) % 12 for x in semis}
                # die natuerliche Variante derselben Stufe entfernen
                if tok in ("b9", "#9"):
                    tensions.discard((root + 2) % 12)
                if tok in ("b13", "#5"):
                    tensions.discard((root + 9) % 12)
                if tok in ("b5", "#11"):
                    avoid.discard((root + 5) % 12)

    return Chord(
        symbol=symbol, root=root, quality=base,
        tones=tones, tensions=sorted(tensions), avoid=sorted(avoid),
    )
