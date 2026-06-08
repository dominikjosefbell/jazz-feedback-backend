"""
standards.py — Kleine Bibliothek bekannter Jazz-Standards mit ihren Changes.

Kernidee der App (siehe CLAUDE.md, Immutable-Entscheidung #3): Die Harmonie
wird NIE blind erraten. Der Nutzer waehlt den Standard → die Changes sind
bekannt → die Engine bewertet nur die *Realisierung* dieser Changes.

Format pro Tune:
  bars: Liste von Takten; pro Takt 1..n Akkordsymbole, die den Takt
        gleichmaessig teilen. Genau das Format von Changes.from_bars().
  beats_per_bar: Taktart-Zaehler (meist 4).
  tempo_hint: grobes Default-Tempo (BPM), nur Vorschlag fuer die UI.

Akkordsymbole muessen vom jazzfb.theory-Parser lesbar sein
(Cmaj7, Dm7, G7, A7b9, F#m7b5, Bb7#11, C7alt, Co7 ...).
"""

from __future__ import annotations
from typing import Optional

# Jeder Eintrag: (Anzeigename, Daten)
STANDARDS: dict[str, dict] = {
    "ii-V-I in C (Uebung)": {
        "beats_per_bar": 4,
        "tempo_hint": 120,
        "bars": [["Dm7"], ["G7"], ["Cmaj7"], ["Cmaj7"]],
    },
    "Autumn Leaves (G-Moll)": {
        "beats_per_bar": 4,
        "tempo_hint": 130,
        "bars": [
            ["Cm7"], ["F7"], ["Bbmaj7"], ["Ebmaj7"],
            ["Am7b5"], ["D7b9"], ["Gm7"], ["Gm7"],
            ["Cm7"], ["F7"], ["Bbmaj7"], ["Ebmaj7"],
            ["Am7b5"], ["D7b9"], ["Gm7"], ["Gm7"],
            ["Am7b5"], ["D7b9"], ["Gm7"], ["Gm7"],
            ["Cm7"], ["F7"], ["Bbmaj7"], ["Ebmaj7"],
            ["Am7b5"], ["D7b9"], ["Gm7", "Gm6"], ["Gm7"],
            ["Am7b5"], ["D7b9"], ["Gm7"], ["Gm7"],
        ],
    },
    "Blue Bossa (C-Moll)": {
        "beats_per_bar": 4,
        "tempo_hint": 150,
        "bars": [
            ["Cm7"], ["Cm7"], ["Fm7"], ["Fm7"],
            ["Dm7b5"], ["G7b9"], ["Cm7"], ["Cm7"],
            ["Ebm7"], ["Ab7"], ["Dbmaj7"], ["Dbmaj7"],
            ["Dm7b5"], ["G7b9"], ["Cm7"], ["Dm7b5", "G7b9"],
        ],
    },
    "So What (D-Dorisch / Eb-Dorisch)": {
        "beats_per_bar": 4,
        "tempo_hint": 130,
        "bars": [
            ["Dm7"], ["Dm7"], ["Dm7"], ["Dm7"],
            ["Dm7"], ["Dm7"], ["Dm7"], ["Dm7"],
            ["Ebm7"], ["Ebm7"], ["Ebm7"], ["Ebm7"],
            ["Dm7"], ["Dm7"], ["Dm7"], ["Dm7"],
        ],
    },
    "Take the A Train (C-Dur)": {
        "beats_per_bar": 4,
        "tempo_hint": 160,
        "bars": [
            ["Cmaj7"], ["Cmaj7"], ["D7#11"], ["D7#11"],
            ["Dm7"], ["G7"], ["Cmaj7"], ["Cmaj7"],
            ["Cmaj7"], ["Cmaj7"], ["D7#11"], ["D7#11"],
            ["Dm7"], ["G7"], ["Cmaj7"], ["Cmaj7"],
        ],
    },
    "Blues in F (12-Takt)": {
        "beats_per_bar": 4,
        "tempo_hint": 130,
        "bars": [
            ["F7"], ["Bb7"], ["F7"], ["F7"],
            ["Bb7"], ["Bb7"], ["F7"], ["D7b9"],
            ["Gm7"], ["C7"], ["F7", "D7b9"], ["Gm7", "C7"],
        ],
    },
    "Blues in Bb (12-Takt)": {
        "beats_per_bar": 4,
        "tempo_hint": 130,
        "bars": [
            ["Bb7"], ["Eb7"], ["Bb7"], ["Bb7"],
            ["Eb7"], ["Eb7"], ["Bb7"], ["G7b9"],
            ["Cm7"], ["F7"], ["Bb7", "G7b9"], ["Cm7", "F7"],
        ],
    },
}


def list_standards() -> list[str]:
    """Anzeigenamen aller bekannten Standards (fuer die UI)."""
    return list(STANDARDS.keys())


def get_standard(name: str) -> Optional[dict]:
    return STANDARDS.get(name)


def parse_manual_changes(text: str) -> list[list[str]]:
    """Freitext-Changes -> bars-Format.

    Konvention: Takte mit '|' trennen, mehrere Akkorde pro Takt mit
    Leerzeichen. Beispiel:
        "Dm7 | G7 | Cmaj7 | Cmaj7"
        "Cmaj7 A7 | Dm7 G7 | Cmaj7 | Cmaj7"
    Leere Takte (Wiederholung) werden uebersprungen.
    """
    bars: list[list[str]] = []
    for raw in text.replace("\n", "|").split("|"):
        chords = raw.split()
        if chords:
            bars.append(chords)
    return bars
