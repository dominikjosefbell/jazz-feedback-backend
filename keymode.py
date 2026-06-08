"""
keymode.py — Tonart-/Modus-Kontext (leichter als volle Changes).

Erlaubt drei Stufen von Harmonie-Kontext (alle optional):
  - volle Changes (Tune)            -> jazzfb.analyze() (woanders)
  - Tonart = Tonika + Modus         -> Klassifikation gegen EINE Tonleiter
  - nur Modus (Tonika unbekannt)    -> Tonika wird aus dem Spiel schaetzbar

Bewusst Tonleiter-basiert (nicht Akkord-basiert): ueber eine ganze Tonart
sind 2/4/6 diatonische Stufen Farbtoene, KEINE Avoid-Noten — der Akkord-Modell-
Ansatz (maj7/m7 …) wuerde sie faelschlich als Avoid markieren.
"""

from __future__ import annotations
from collections import Counter
from typing import Optional

from jazzfb.theory import NOTE_TO_PC, pc_name

# Modus -> Intervalle (Halbtoene ueber der Tonika), 7-stufig.
MODES: dict[str, list[int]] = {
    "major":          [0, 2, 4, 5, 7, 9, 11],   # Ionisch
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "phrygian":       [0, 1, 3, 5, 7, 8, 10],
    "lydian":         [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
    "minor":          [0, 2, 3, 5, 7, 8, 10],   # Aeolisch
    "locrian":        [0, 1, 3, 5, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":  [0, 2, 3, 5, 7, 9, 11],
}

MODE_LABELS: dict[str, str] = {
    "major": "Dur (Ionisch)", "dorian": "Dorisch", "phrygian": "Phrygisch",
    "lydian": "Lydisch", "mixolydian": "Mixolydisch", "minor": "Moll (Aeolisch)",
    "locrian": "Lokrisch", "harmonic_minor": "Harmonisch Moll",
    "melodic_minor": "Melodisch Moll",
}


def list_modes() -> list[dict]:
    """Fuer die UI: [{value, label}, …]."""
    return [{"value": k, "label": MODE_LABELS[k]} for k in MODES]


def parse_tonic(text: Optional[str]) -> Optional[int]:
    """'C', 'Bb', 'F#' … -> Pitchclass. Leer/unbekannt -> None."""
    if not text:
        return None
    return NOTE_TO_PC.get(text.strip().upper())


def scale_pcs(tonic_pc: int, mode: str) -> set[int]:
    return {(tonic_pc + i) % 12 for i in MODES[mode]}


def stable_pcs(tonic_pc: int, mode: str) -> set[int]:
    """Stabile Toene = Tonika-Septakkord (Stufen 1,3,5,7)."""
    iv = MODES[mode]
    return {(tonic_pc + iv[i]) % 12 for i in (0, 2, 4, 6)}


def classify(pc: int, tonic_pc: int, mode: str) -> str:
    """chord_tone (stabil 1/3/5/7) | tension (sonstige Tonleiterstufe) | chromatic."""
    if pc in stable_pcs(tonic_pc, mode):
        return "chord_tone"
    if pc in scale_pcs(tonic_pc, mode):
        return "tension"
    return "chromatic"


def infer_tonic(pcs: Counter, mode: str) -> int:
    """Schaetzt die Tonika bei bekanntem Modus: waehlt die Tonika, deren
    Tonleiter am meisten gespielte Toene abdeckt (Stabiltoene zaehlen extra).
    Robust, weil nur EIN Modus getestet wird — nicht blind Modus+Tonika."""
    best_pc, best_score = 0, -1.0
    total = sum(pcs.values()) or 1
    for cand in range(12):
        sc = scale_pcs(cand, mode)
        st = stable_pcs(cand, mode)
        in_scale = sum(w for p, w in pcs.items() if p in sc)
        on_stable = sum(w for p, w in pcs.items() if p in st)
        on_tonic = pcs.get(cand, 0)
        score = in_scale / total + 0.25 * (on_stable / total) + 0.1 * (on_tonic / total)
        if score > best_score:
            best_score, best_pc = score, cand
    return best_pc


def key_label(tonic_pc: int, mode: str) -> str:
    return f"{pc_name(tonic_pc)} {MODE_LABELS.get(mode, mode)}"
