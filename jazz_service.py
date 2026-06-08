"""
jazz_service.py — Orchestrierung zwischen Upload, jazzfb-Engine und LLM.

Ersetzt die alte blinde Harmonie-Erkennung (midi_analyzer.identify_jazz_chord
& Co.) durch den klaren Weg der jazzfb-Engine:

    MIDI  ->  jazzfb.Note-Events
              + bekannte Changes (Nutzer waehlt den Standard)
              + Beat-Raster (Tempo aus MIDI oder Vorgabe)
              ->  jazzfb.analyze()  ->  strukturierter Report
              ->  rule_based_summary()  (ohne LLM)
              ->  facts_for_llm()  ->  Apertus (Prosa/Scores in main.py)

Die Engine bleibt LLM-unabhaengig; dieses Modul kennt kein Apertus/Claude,
es liefert nur den Report + einen kompakten Faktentext fuer den Prompt.
"""

from __future__ import annotations
from typing import Optional

from jazzfb import Note, BeatGrid, Changes, analyze, rule_based_summary, from_midi


# --- Tempo aus MIDI lesen (nur fuer den Default-Vorschlag) ------------------

def tempo_from_midi(path: str) -> Optional[float]:
    """Liest das erste set_tempo aus einer MIDI-Datei (mido). None, wenn keins."""
    try:
        import mido
    except ImportError:
        return None
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return round(mido.tempo2bpm(msg.tempo), 1)
    return None


# --- Changes bauen ----------------------------------------------------------

def changes_from_bars(bars: list[list[str]], beats_per_bar: int = 4) -> Changes:
    return Changes.from_bars(bars, beats_per_bar=beats_per_bar)


# --- Haupt-Einstieg ---------------------------------------------------------

def analyze_midi_with_changes(
    midi_path: str,
    bars: list[list[str]],
    beats_per_bar: int = 4,
    bpm: Optional[float] = None,
    downbeat: Optional[float] = None,
) -> dict:
    """Laedt die MIDI-Datei, baut Raster + Changes und laesst jazzfb laufen.

    bpm/downbeat optional: bpm sonst aus MIDI bzw. 120; downbeat sonst der
    Onset der ersten Note (Annahme: Aufnahme beginnt auf Beat 1).

    Rueckgabe (JSON-faehig):
      { ok, report, summary, used: {bpm, downbeat, beats_per_bar, n_notes} }
    oder { ok: False, error }.
    """
    notes = from_midi(midi_path)
    if not notes:
        return {"ok": False, "error": "Keine Noten in der Datei gefunden."}

    if bpm is None:
        bpm = tempo_from_midi(midi_path) or 120.0
    if downbeat is None:
        downbeat = min(n.onset for n in notes)

    grid = BeatGrid(bpm=float(bpm), start=float(downbeat),
                    beats_per_bar=int(beats_per_bar))
    changes = changes_from_bars(bars, beats_per_bar=beats_per_bar)

    report = analyze(notes, grid, changes)
    summary = rule_based_summary(report)

    return {
        "ok": True,
        "report": report,
        "summary": summary,
        "used": {
            "bpm": round(float(bpm), 1),
            "downbeat": round(float(downbeat), 3),
            "beats_per_bar": int(beats_per_bar),
            "n_notes": len(notes),
        },
    }


def analyze_notes_with_changes(
    notes: list[Note],
    bars: list[list[str]],
    beats_per_bar: int = 4,
    bpm: float = 120.0,
    downbeat: Optional[float] = None,
) -> dict:
    """Wie oben, aber fuer bereits geladene Note-Events (z.B. aus Basic Pitch
    in Slice 2, oder fuer Tests). Keine MIDI-Datei noetig."""
    if not notes:
        return {"ok": False, "error": "Keine Noten uebergeben."}
    if downbeat is None:
        downbeat = min(n.onset for n in notes)
    grid = BeatGrid(bpm=float(bpm), start=float(downbeat),
                    beats_per_bar=int(beats_per_bar))
    changes = changes_from_bars(bars, beats_per_bar=beats_per_bar)
    report = analyze(notes, grid, changes)
    return {
        "ok": True,
        "report": report,
        "summary": rule_based_summary(report),
        "used": {"bpm": round(float(bpm), 1),
                 "downbeat": round(float(downbeat), 3),
                 "beats_per_bar": int(beats_per_bar),
                 "n_notes": len(notes)},
    }


# --- Kompakter Faktentext fuer den LLM-Prompt -------------------------------

def facts_for_llm(report: dict, summary: list[str], tune_name: str) -> str:
    """Verdichtet den Report zu einem knappen, belastbaren Faktentext.
    Diese Fakten stammen aus Regeln (Musiktheorie gegen bekannte Changes),
    nicht aus geratener Harmonie — der LLM soll sie interpretieren, nicht neu
    erfinden."""
    line = report["line"]
    voic = report["voicings"]
    vl = report["voice_leading"]
    tf = report["time_feel"]
    contour = report.get("contour", {})
    dist = line["distribution"]

    parts = [f"Tune/Changes: {tune_name}"]
    parts.append(
        f"Linie: {line['n_notes']} Toene — "
        f"{round(dist['chord_tone']*100)}% Akkordtoene, "
        f"{round(dist['tension']*100)}% Tensions, "
        f"{round(dist['chromatic']*100)}% chromatisch, "
        f"{round(dist['avoid']*100)}% Avoid."
    )
    if line["chord_tones_on_strong_beats"] is not None:
        parts.append(
            "Akkordtoene auf betonten Zeiten: "
            f"{round(line['chord_tones_on_strong_beats']*100)}%."
        )
    if line["avoid_notes_on_strong_beats"]:
        spots = ", ".join(
            f"{a['note']} auf {a['chord']} (Takt {a['bar']})"
            for a in line["avoid_notes_on_strong_beats"][:4]
        )
        parts.append(f"Avoid-Noten auf betonten Zeiten: {spots}.")
    if voic["n_voicings"]:
        parts.append(
            f"Voicings: {voic['n_voicings']} erkannt, "
            f"{round(voic['rootless_ratio']*100)}% rootless, "
            f"Leitton-Abdeckung {round(voic['guide_tone_coverage']*100)}%, "
            f"Ø {voic['avg_tensions_per_voicing']} Tensions/Voicing."
        )
    if vl.get("top_voice_avg_leap_semitones") is not None:
        parts.append(
            f"Stimmfuehrung Oberstimme: {vl['smoothness_comment']} "
            f"(Ø {vl['top_voice_avg_leap_semitones']} Halbtoene, "
            f"{round(vl['top_voice_stepwise_ratio']*100)}% schrittweise)."
        )
    if tf.get("swing_ratio"):
        parts.append(f"Time-Feel: {tf['feel_comment']}; {tf['timing_comment']}.")
    if contour:
        parts.append(
            f"Ambitus: {contour.get('pitch_range_semitones')} Halbtoene; "
            f"Dichte: {contour.get('notes_per_second')} Toene/Sek; "
            f"Dynamik-Range (Velocity): {contour.get('velocity_range')}."
        )
    parts.append("Regel-Zusammenfassung:")
    parts.extend("  - " + s for s in summary)
    return "\n".join(parts)
