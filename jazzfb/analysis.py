"""
analysis.py — Die analytische Schicht. Erzeugt aus getrennten Noten +
Changes + Beat-Raster eine strukturierte, JSON-faehige Auswertung.

Bewusst statistisch/strukturell statt per-Note, damit Transkriptionsfehler
das Ergebnis nicht kippen.
"""

from __future__ import annotations
from .core import Note, BeatGrid, Changes
from .separation import Separated, Cluster
from .theory import pc_name, Chord


def _chord_for(note: Note, grid: BeatGrid, changes: Changes):
    bar, beat = grid.position(note.onset)
    span = changes.chord_at(bar, beat)
    return span.chord if span else None


# --- Linie: Akkordton-/Tension-Nutzung -------------------------------------

def analyze_line(line: list[Note], grid: BeatGrid, changes: Changes) -> dict:
    counts = {"chord_tone": 0, "tension": 0, "avoid": 0, "chromatic": 0}
    strong_total = strong_chord_tone = 0
    avoid_on_strong = []
    detail = []
    for n in line:
        ch = _chord_for(n, grid, changes)
        if ch is None:
            continue
        cls = ch.classify(n.pc)
        counts[cls] += 1
        strong = grid.is_strong(n.onset)
        if strong:
            strong_total += 1
            if cls == "chord_tone":
                strong_chord_tone += 1
            if cls == "avoid":
                bar, beat = grid.position(n.onset)
                avoid_on_strong.append(dict(bar=bar, beat=round(beat, 2),
                                            note=pc_name(n.pc), chord=ch.symbol))
        detail.append(dict(t=round(n.onset, 3), note=pc_name(n.pc),
                           chord=ch.symbol, role=cls, strong=strong))
    total = sum(counts.values()) or 1
    return {
        "n_notes": sum(counts.values()),
        "distribution": {k: round(v / total, 3) for k, v in counts.items()},
        "counts": counts,
        "chord_tones_on_strong_beats": (
            round(strong_chord_tone / strong_total, 3) if strong_total else None),
        "avoid_notes_on_strong_beats": avoid_on_strong,
        "detail": detail,
    }


# --- Voicings: Realisierung der Changes -------------------------------------

def analyze_voicings(clusters: list[Cluster], grid: BeatGrid, changes: Changes) -> dict:
    out = []
    for c in clusters:
        if len(c.pitches) < 3:
            continue                      # Einzeltoene/Zweiklaenge hier ueberspringen
        bar, beat = grid.position(c.onset)
        span = changes.chord_at(bar, beat)
        if span is None:
            continue
        ch = span.chord
        present = c.pcs
        tones_present = sorted(present & set(ch.tones))
        tens_present = sorted(present & set(ch.tensions))
        outside = sorted(present - set(ch.tones) - set(ch.tensions))
        root_omitted = ch.root not in present
        guide = set(ch.guide_tones())
        guide_present = guide.issubset(present)
        out.append(dict(
            bar=bar, beat=round(beat, 2), chord=ch.symbol,
            pitches=c.pitches,
            chord_tones=[pc_name(p) for p in tones_present],
            tensions=[pc_name(p) for p in tens_present],
            outside=[pc_name(p) for p in outside],
            rootless=root_omitted,
            guide_tones_present=guide_present,
            size=len(c.pitches),
        ))
    n = len(out) or 1
    return {
        "n_voicings": len(out),
        "rootless_ratio": round(sum(v["rootless"] for v in out) / n, 3),
        "guide_tone_coverage": round(sum(v["guide_tones_present"] for v in out) / n, 3),
        "avg_tensions_per_voicing": round(
            sum(len(v["tensions"]) for v in out) / n, 2),
        "voicings": out,
    }


# --- Voice-Leading: Leittonlinien zwischen Voicings -------------------------

def analyze_voice_leading(voicing_report: dict) -> dict:
    vs = voicing_report["voicings"]
    if len(vs) < 2:
        return {"top_voice_avg_leap": None, "smoothness_comment": "zu wenige Voicings"}
    tops = [max(v["pitches"]) for v in vs]
    leaps = [abs(tops[i + 1] - tops[i]) for i in range(len(tops) - 1)]
    avg = sum(leaps) / len(leaps)
    by_step = sum(1 for l in leaps if l <= 2) / len(leaps)
    return {
        "top_voice_avg_leap_semitones": round(avg, 2),
        "top_voice_stepwise_ratio": round(by_step, 3),
        "smoothness_comment": ("glatt/stimmfuehrungsorientiert" if avg <= 3
                               else "sprunghaft — Voicings springen im Register"),
    }


# --- Time-Feel: Swing-Ratio und Lage zum Beat ------------------------------

def analyze_time_feel(line: list[Note], grid: BeatGrid) -> dict:
    if len(line) < 4:
        return {"swing_ratio": None, "timing_bias_beats": None,
                "comment": "zu wenig Linienmaterial"}
    spb = grid.spb
    onsets = sorted(n.onset for n in line)
    first_beat = int((onsets[0] - grid.start) / spb)
    last_beat = int((onsets[-1] - grid.start) / spb) + 1

    # Lage zum Beat: nur Toene NAHE an einem Beat zaehlen (die "auf der Zeit"
    # gemeint sind) — Offbeats wuerden den Wert verfaelschen.
    near = [grid.beat_phase(o) for o in onsets if abs(grid.beat_phase(o)) < 0.2]
    bias = sum(near) / len(near) if near else 0.0

    # Swing-Ratio, am Raster verankert: pro Beat den Offbeat-Ton suchen
    # (Phase 0.35..0.8 des Beats); Verhaeltnis erste:zweite Achtel.
    ratios = []
    for k in range(first_beat, last_beat):
        beat_t = grid.start + k * spb
        for o in onsets:
            frac = (o - beat_t) / spb
            if 0.35 < frac < 0.8:
                ratios.append(frac / (1 - frac))
                break
    swing = sorted(ratios)[len(ratios) // 2] if ratios else None   # Median
    if swing is None:
        feel = "unbestimmt"
    elif swing >= 1.4:
        feel = f"deutlicher Swing (~{swing:.1f}:1, Richtung Triolen)"
    elif swing >= 1.15:
        feel = f"leichter Swing (~{swing:.2f}:1)"
    else:
        feel = "even/straight (kaum Swing)"
    return {
        "swing_ratio": round(swing, 3) if swing else None,
        "timing_bias_beats": round(bias, 3),
        "timing_comment": ("eher hinter der Zeit (laid back)" if bias > 0.02
                           else "eher vor der Zeit (pushing)" if bias < -0.02
                           else "auf der Zeit"),
        "feel_comment": feel,
    }


# --- Dynamik / Register / Dichte -------------------------------------------

def analyze_contour(notes: list[Note], grid: BeatGrid) -> dict:
    if not notes:
        return {}
    pitches = [n.pitch for n in notes]
    vels = [n.velocity for n in notes]
    span_s = max(n.offset for n in notes) - min(n.onset for n in notes)
    return {
        "pitch_range_semitones": max(pitches) - min(pitches),
        "lowest": min(pitches), "highest": max(pitches),
        "velocity_mean": round(sum(vels) / len(vels), 1),
        "velocity_range": max(vels) - min(vels),
        "notes_per_second": round(len(notes) / span_s, 2) if span_s > 0 else None,
    }
