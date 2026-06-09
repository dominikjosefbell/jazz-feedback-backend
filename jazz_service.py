"""
jazz_service.py — Orchestrierung zwischen Upload, jazzfb-Engine und LLM.

Harmonie-Kontext ist OPTIONAL und in drei Stufen moeglich:
  - "changes": Tune/Changes vorgegeben  -> volle jazzfb.analyze()
  - "key":     Tonart (Tonika+Modus, oder nur Modus -> Tonika geschaetzt)
               -> Klassifikation der Linie gegen EINE Tonleiter
  - "none":    kein Kontext -> nur kontextfreie Analyse
               (Time-Feel, Kontur, Stimmfuehrung, Linienform)

Time-Feel, Kontur und Oberstimmen-Stimmfuehrung brauchen KEINE Harmonie und
laufen in jeder Stufe. Nur die harmonische Bewertung skaliert mit dem Kontext.

Die jazzfb-Engine bleibt LLM-unabhaengig; dieses Modul kennt kein Apertus/Claude,
es liefert den Report + einen kompakten Faktentext fuer den Prompt.
"""

from __future__ import annotations
from collections import Counter
from typing import Optional

from jazzfb import Note, BeatGrid, Changes, analyze, rule_based_summary, from_midi
from jazzfb.core import from_basic_pitch
from jazzfb.separation import separate
from jazzfb.analysis import analyze_time_feel, analyze_contour, analyze_voice_leading
from jazzfb.theory import pc_name
import keymode
import standards


# --- Tempo aus MIDI lesen (nur fuer den Default-Vorschlag) ------------------

def tempo_from_midi(path: str) -> Optional[float]:
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


# --- Kontext aufloesen (Precedence: manuelle Changes > Tune > Tonart > nichts)

def resolve_context(tune: Optional[str], manual_changes: Optional[str],
                    key_tonic: Optional[str], key_mode: Optional[str]) -> dict:
    if manual_changes and manual_changes.strip():
        bars = standards.parse_manual_changes(manual_changes)
        return {"kind": "changes", "bars": bars, "beats_per_bar": 4,
                "tempo_hint": None, "label": "Eigene Changes"}
    std = standards.get_standard(tune) if tune else None
    if std:
        return {"kind": "changes", "bars": std["bars"],
                "beats_per_bar": std.get("beats_per_bar", 4),
                "tempo_hint": std.get("tempo_hint"), "label": tune}
    if key_mode and key_mode in keymode.MODES:
        tonic_pc = keymode.parse_tonic(key_tonic)
        return {"kind": "key", "beats_per_bar": 4, "tempo_hint": None,
                "mode": key_mode, "tonic_pc": tonic_pc,
                "tonic_known": tonic_pc is not None}
    return {"kind": "none", "beats_per_bar": 4, "tempo_hint": None,
            "label": "Ohne Harmonie-Kontext"}


# --- kontextfreie Bausteine -------------------------------------------------

def _voice_leading_from_clusters(clusters) -> dict:
    """Oberstimmen-Stimmfuehrung — braucht keine Harmonie, nur die Voicing-Toene."""
    vs = [{"pitches": c.pitches} for c in clusters if len(c.pitches) >= 3]
    return analyze_voice_leading({"voicings": vs})


def _line_no_harmony(line: list[Note]) -> dict:
    if not line:
        return {"n_notes": 0, "distribution": None,
                "chord_tones_on_strong_beats": None,
                "avoid_notes_on_strong_beats": []}
    pitches = [n.pitch for n in line]
    iv = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    step = sum(1 for d in iv if d <= 2) / len(iv) if iv else 0.0
    return {
        "n_notes": len(line),
        "distribution": None,
        "chord_tones_on_strong_beats": None,
        "avoid_notes_on_strong_beats": [],
        "stepwise_ratio": round(step, 3),
        "avg_interval_semitones": round(sum(iv) / len(iv), 2) if iv else None,
    }


def _line_against_scale(line: list[Note], grid: BeatGrid,
                        tonic_pc: int, mode: str) -> dict:
    counts = {"chord_tone": 0, "tension": 0, "avoid": 0, "chromatic": 0}
    strong_total = strong_stable = 0
    chromatic_on_strong = []
    for n in line:
        cls = keymode.classify(n.pc, tonic_pc, mode)
        counts[cls] += 1
        if grid.is_strong(n.onset):
            strong_total += 1
            if cls == "chord_tone":
                strong_stable += 1
            elif cls == "chromatic":
                bar, beat = grid.position(n.onset)
                chromatic_on_strong.append(
                    dict(bar=bar, beat=round(beat, 2), note=pc_name(n.pc),
                         chord=keymode.key_label(tonic_pc, mode)))
    total = sum(counts.values()) or 1
    return {
        "n_notes": sum(counts.values()),
        "distribution": {k: round(v / total, 3) for k, v in counts.items()},
        "counts": counts,
        "chord_tones_on_strong_beats": (
            round(strong_stable / strong_total, 3) if strong_total else None),
        # chromatische (tonleiterfremde) Toene auf betonten Zeiten — Analogon
        # zu Avoid-Noten, hier aber "ausserhalb der Tonleiter".
        "avoid_notes_on_strong_beats": chromatic_on_strong,
    }


def _comp_in_scale_ratio(clusters, tonic_pc: int, mode: str) -> Optional[float]:
    sc = keymode.scale_pcs(tonic_pc, mode)
    voic = [c for c in clusters if len(c.pitches) >= 3]
    if not voic:
        return None
    total = sum(len(c.pcs) for c in voic)
    inside = sum(len(c.pcs & sc) for c in voic)
    return round(inside / total, 3) if total else None


# --- Haupt-Dispatcher -------------------------------------------------------

def _notes_view(sep) -> list[dict]:
    """Flache Notenliste fuer die Piano-Roll-Darstellung (Transparenz/Pruefung).
    voice = 'line' (Melodie) oder 'comp' (Begleit-Voicing) — zeigt zugleich,
    wie die Rollen-Trennung entschieden hat."""
    out = []
    for n in sep.line:
        out.append({"on": round(n.onset, 3), "off": round(n.offset, 3),
                    "p": n.pitch, "voice": "line"})
    for c in sep.clusters:
        for n in c.notes:
            out.append({"on": round(n.onset, 3), "off": round(n.offset, 3),
                        "p": n.pitch, "voice": "comp"})
    return sorted(out, key=lambda x: (x["on"], x["p"]))


def analyze_recording(notes: list[Note], grid: BeatGrid, context: dict) -> dict:
    kind = context.get("kind", "none")
    sep = separate(notes)

    if kind == "changes":
        changes = Changes.from_bars(context["bars"], beats_per_bar=grid.beats_per_bar)
        report = analyze(notes, grid, changes)        # volle, bewaehrte Analyse
        report["context"] = {"kind": "changes", "label": context.get("label", "Changes")}
        report["changes_view"] = [
            {"bar": s.bar, "beat": round(s.beat, 2), "beats": s.beats, "symbol": s.symbol}
            for s in changes.spans]
    else:
        meta = {
            "n_notes": len(notes), "bpm": grid.bpm,
            "beats_per_bar": grid.beats_per_bar,
            "line_role": sep.line_role, "comp_role": sep.comp_role,
            "n_line_notes": len(sep.line), "n_clusters": len(sep.clusters),
        }
        report = {
            "meta": meta,
            "voicings": {"n_voicings": 0,
                         "comp_clusters": len([c for c in sep.clusters if len(c.pitches) >= 3])},
            "voice_leading": _voice_leading_from_clusters(sep.clusters),
            "time_feel": analyze_time_feel(sep.line, grid),
            "contour": analyze_contour(notes, grid),
        }
        if kind == "key":
            tonic_pc = context.get("tonic_pc")
            mode = context["mode"]
            tonic_known = context.get("tonic_known", tonic_pc is not None)
            if tonic_pc is None:
                pcs = Counter(n.pc for n in sep.line) or Counter(n.pc for n in notes)
                tonic_pc = keymode.infer_tonic(pcs, mode)
            report["line"] = _line_against_scale(sep.line, grid, tonic_pc, mode)
            report["voicings"]["comp_in_scale_ratio"] = _comp_in_scale_ratio(
                sep.clusters, tonic_pc, mode)
            report["context"] = {
                "kind": "key", "label": keymode.key_label(tonic_pc, mode),
                "tonic": pc_name(tonic_pc), "mode": mode, "tonic_known": tonic_known,
            }
        else:  # kind == "none"
            report["line"] = _line_no_harmony(sep.line)
            report["context"] = {"kind": "none", "label": "Ohne Harmonie-Kontext"}

    # Gemeinsam: Raster + Notenliste fuer die Piano-Roll.
    report["grid"] = {
        "bpm": grid.bpm, "downbeat": round(grid.start, 3),
        "beats_per_bar": grid.beats_per_bar,
        "t_start": round(min((n.onset for n in notes), default=0.0), 3),
        "t_end": round(max((n.offset for n in notes), default=0.0), 3),
    }
    report["notes_view"] = _notes_view(sep)
    return report


def analyze_midi(midi_path: str, context: dict,
                 beats_per_bar: Optional[int] = None,
                 bpm: Optional[float] = None,
                 downbeat: Optional[float] = None) -> dict:
    notes = from_midi(midi_path)
    if not notes:
        return {"ok": False, "error": "Keine Noten in der Datei gefunden."}
    if bpm is None:
        bpm = tempo_from_midi(midi_path) or context.get("tempo_hint") or 120.0
    if downbeat is None:
        downbeat = min(n.onset for n in notes)
    bpb = int(beats_per_bar or context.get("beats_per_bar", 4))
    grid = BeatGrid(bpm=float(bpm), start=float(downbeat), beats_per_bar=bpb)

    report = analyze_recording(notes, grid, context)
    return {
        "ok": True,
        "report": report,
        "summary": summarize(report),
        "used": {"bpm": round(float(bpm), 1), "downbeat": round(float(downbeat), 3),
                 "beats_per_bar": bpb, "n_notes": len(notes),
                 "context": report.get("context", {})},
    }


def analyze_notes(note_events, context: dict,
                  beats_per_bar: Optional[int] = None,
                  bpm: Optional[float] = None,
                  downbeat: Optional[float] = None) -> dict:
    """Analyse aus rohen Note-Events (z.B. Spotify Basic Pitch im Browser).
    note_events: Liste von [start_s, end_s, pitch_midi, amplitude]."""
    notes = from_basic_pitch(note_events)
    if not notes:
        return {"ok": False, "error": "Keine Noten in der Transkription."}
    # Audio liefert kein Tempo -> Vorgabe/Standard. Genaues bpm/downbeat ist hier
    # besonders wichtig (siehe Hinweis in der UI).
    if bpm is None:
        bpm = context.get("tempo_hint") or 120.0
    if downbeat is None:
        downbeat = min(n.onset for n in notes)
    bpb = int(beats_per_bar or context.get("beats_per_bar", 4))
    grid = BeatGrid(bpm=float(bpm), start=float(downbeat), beats_per_bar=bpb)

    report = analyze_recording(notes, grid, context)
    return {
        "ok": True,
        "report": report,
        "summary": summarize(report),
        "used": {"bpm": round(float(bpm), 1), "downbeat": round(float(downbeat), 3),
                 "beats_per_bar": bpb, "n_notes": len(notes),
                 "source": "audio", "context": report.get("context", {})},
    }


# Rueckwaerts-kompatibler Helfer (Changes direkt als bars).
def analyze_midi_with_changes(midi_path: str, bars: list[list[str]],
                              beats_per_bar: int = 4,
                              bpm: Optional[float] = None) -> dict:
    ctx = {"kind": "changes", "bars": bars, "beats_per_bar": beats_per_bar,
           "label": "Changes"}
    return analyze_midi(midi_path, ctx, beats_per_bar=beats_per_bar, bpm=bpm)


# --- Zusammenfassung + LLM-Fakten (kontextabhaengig) ------------------------

def summarize(report: dict) -> list[str]:
    ctx = report.get("context", {})
    kind = ctx.get("kind", "none")
    if kind == "changes":
        return rule_based_summary(report)

    out = []
    line = report.get("line", {})
    if kind == "key":
        if line.get("chord_tones_on_strong_beats") is not None:
            out.append(
                f"Stabile Tonleitertoene (1/3/5/7) auf betonten Zeiten: "
                f"{round(line['chord_tones_on_strong_beats'] * 100)}%.")
        dist = line.get("distribution") or {}
        if dist:
            out.append(
                f"Linie gegen {ctx.get('label', 'Tonart')}: "
                f"{round(dist['chord_tone'] * 100)}% stabil, "
                f"{round(dist['tension'] * 100)}% Tonleiter-Farbtoene, "
                f"{round(dist['chromatic'] * 100)}% chromatisch (ausserhalb).")
        av = line.get("avoid_notes_on_strong_beats") or []
        if av:
            spots = ", ".join(f"{a['note']} (Takt {a['bar']})" for a in av[:4])
            out.append(f"Tonleiterfremde Toene auf betonten Zeiten: {spots}"
                       + (" …" if len(av) > 4 else "") + ".")
        r = report.get("voicings", {}).get("comp_in_scale_ratio")
        if r is not None:
            out.append(f"Begleit-Voicings: {round(r * 100)}% ihrer Toene liegen in der Tonleiter.")
    elif kind == "none":
        out.append(
            f"Ohne Harmonie-Kontext: {line.get('n_notes', 0)} Linientoene, "
            f"{round((line.get('stepwise_ratio') or 0) * 100)}% schrittweise gefuehrt.")

    vl = report.get("voice_leading", {})
    if vl.get("top_voice_avg_leap_semitones") is not None:
        out.append(f"Stimmfuehrung Oberstimme: {vl['smoothness_comment']} "
                   f"(Ø {vl['top_voice_avg_leap_semitones']} Halbtoene, "
                   f"{round(vl['top_voice_stepwise_ratio'] * 100)}% schrittweise).")
    tf = report.get("time_feel", {})
    if tf.get("swing_ratio"):
        out.append(f"Time-Feel: {tf['feel_comment']}; {tf['timing_comment']}.")
    cont = report.get("contour", {})
    if cont:
        out.append(f"Ambitus {cont.get('pitch_range_semitones')} Halbtoene, "
                   f"Dichte {cont.get('notes_per_second')} Toene/Sek.")
    return out


def facts_for_llm(report: dict, summary: list[str], label: str) -> str:
    """Kompakter Faktentext fuer den LLM-Prompt — kontextabhaengig."""
    ctx = report.get("context", {})
    kind = ctx.get("kind", "none")
    line = report.get("line", {})
    tf = report.get("time_feel", {})
    cont = report.get("contour", {})
    vl = report.get("voice_leading", {})

    if kind == "changes":
        parts = [f"Kontext: volle Changes — {label}."]
        dist = line.get("distribution", {})
        parts.append(
            f"Linie: {line.get('n_notes', 0)} Toene — "
            f"{round(dist.get('chord_tone', 0) * 100)}% Akkordtoene, "
            f"{round(dist.get('tension', 0) * 100)}% Tensions, "
            f"{round(dist.get('chromatic', 0) * 100)}% chromatisch, "
            f"{round(dist.get('avoid', 0) * 100)}% Avoid.")
        if line.get("chord_tones_on_strong_beats") is not None:
            parts.append("Akkordtoene auf betonten Zeiten: "
                         f"{round(line['chord_tones_on_strong_beats'] * 100)}%.")
        if line.get("avoid_notes_on_strong_beats"):
            spots = ", ".join(f"{a['note']} auf {a['chord']} (Takt {a['bar']})"
                              for a in line["avoid_notes_on_strong_beats"][:4])
            parts.append(f"Avoid-Noten auf betonten Zeiten: {spots}.")
        voic = report.get("voicings", {})
        if voic.get("n_voicings"):
            parts.append(
                f"Voicings: {voic['n_voicings']} erkannt, "
                f"{round(voic['rootless_ratio'] * 100)}% rootless, "
                f"Leitton-Abdeckung {round(voic['guide_tone_coverage'] * 100)}%, "
                f"Ø {voic['avg_tensions_per_voicing']} Tensions/Voicing.")
    elif kind == "key":
        parts = [f"Kontext: EINE Tonart/Tonleiter — {label}"
                 + ("" if ctx.get("tonic_known") else " (Tonika aus dem Spiel geschaetzt)") + "."]
        dist = line.get("distribution") or {}
        if dist:
            parts.append(
                f"Linie gegen die Tonleiter: {round(dist['chord_tone'] * 100)}% stabile "
                f"Toene (1/3/5/7), {round(dist['tension'] * 100)}% sonstige Tonleiterstufen, "
                f"{round(dist['chromatic'] * 100)}% chromatisch (ausserhalb).")
        if line.get("chord_tones_on_strong_beats") is not None:
            parts.append("Stabile Toene auf betonten Zeiten: "
                         f"{round(line['chord_tones_on_strong_beats'] * 100)}%.")
        r = report.get("voicings", {}).get("comp_in_scale_ratio")
        if r is not None:
            parts.append(f"Begleit-Voicings: {round(r * 100)}% der Toene in der Tonleiter.")
        parts.append("Hinweis: Bewertung auf Tonart-Ebene, nicht pro Akkord.")
    else:
        parts = ["Kontext: KEINER (keine Changes/Tonart angegeben)."]
        parts.append(f"Linie: {line.get('n_notes', 0)} Toene, "
                     f"{round((line.get('stepwise_ratio') or 0) * 100)}% schrittweise, "
                     f"Ø Intervall {line.get('avg_interval_semitones')} Halbtoene.")
        parts.append("Hinweis: keine harmonische Bewertung moeglich — nur Time-Feel, "
                     "Stimmfuehrung, Kontur.")

    if vl.get("top_voice_avg_leap_semitones") is not None:
        parts.append(f"Stimmfuehrung Oberstimme: {vl['smoothness_comment']} "
                     f"(Ø {vl['top_voice_avg_leap_semitones']} HT, "
                     f"{round(vl['top_voice_stepwise_ratio'] * 100)}% schrittweise).")
    if tf.get("swing_ratio"):
        parts.append(f"Time-Feel: {tf['feel_comment']}; {tf['timing_comment']}.")
    if cont:
        parts.append(f"Ambitus: {cont.get('pitch_range_semitones')} Halbtoene; "
                     f"Dichte: {cont.get('notes_per_second')} Toene/Sek; "
                     f"Dynamik-Range (Velocity): {cont.get('velocity_range')}.")
    parts.append("Regel-Zusammenfassung:")
    parts.extend("  - " + s for s in summary)
    return "\n".join(parts)
