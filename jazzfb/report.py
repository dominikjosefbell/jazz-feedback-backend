"""
report.py — Orchestrierung. analyze() fuehrt alle Schichten zusammen zu einer
strukturierten, JSON-faehigen Auswertung (genau die Datenstruktur, auf der
Regeln UND ein LLM-Feedback aufsetzen).
"""

from __future__ import annotations
import json
from .core import Note, BeatGrid, Changes
from .separation import separate
from .analysis import (analyze_line, analyze_voicings, analyze_voice_leading,
                       analyze_time_feel, analyze_contour)


def analyze(notes: list[Note], grid: BeatGrid, changes: Changes) -> dict:
    sep = separate(notes)
    voic = analyze_voicings(sep.clusters, grid, changes)
    report = {
        "meta": {
            "n_notes": len(notes),
            "bpm": grid.bpm,
            "beats_per_bar": grid.beats_per_bar,
            "line_role": sep.line_role,
            "comp_role": sep.comp_role,
            "n_line_notes": len(sep.line),
            "n_clusters": len(sep.clusters),
        },
        "line": analyze_line(sep.line, grid, changes),
        "voicings": voic,
        "voice_leading": analyze_voice_leading(voic),
        "time_feel": analyze_time_feel(sep.line, grid),
        "contour": analyze_contour(notes, grid),
    }
    return report


# --- Regelbasierte Kurz-Zusammenfassung (ohne LLM) -------------------------

def rule_based_summary(r: dict) -> list[str]:
    out = []
    line = r["line"]
    if line["chord_tones_on_strong_beats"] is not None:
        pct = round(line["chord_tones_on_strong_beats"] * 100)
        if pct >= 65:
            out.append(f"Starke Akkordton-Verankerung auf betonten Zeiten ({pct}%) "
                       "— die Linie sitzt klar in den Changes.")
        elif pct >= 40:
            out.append(f"Mittlere Akkordton-Bindung auf betonten Zeiten ({pct}%) — "
                       "Raum, Zieltoene bewusster auf die Wechsel zu setzen.")
        else:
            out.append(f"Wenig Akkordton-Verankerung ({pct}% auf betonten Zeiten) — "
                       "die Linie wirkt vermutlich schwebend/ausserhalb der Changes.")
    av = line["avoid_notes_on_strong_beats"]
    if av:
        spots = ", ".join(f"{a['note']} auf {a['chord']} (Takt {a['bar']})" for a in av[:3])
        out.append(f"Avoid-Noten auf betonten Zeiten: {spots}"
                   + (" …" if len(av) > 3 else "") + ".")
    dist = line["distribution"]
    out.append(f"Linien-Mix: {round(dist['chord_tone']*100)}% Akkordtoene, "
               f"{round(dist['tension']*100)}% Tensions, "
               f"{round(dist['chromatic']*100)}% chromatisch.")
    v = r["voicings"]
    if v["n_voicings"]:
        out.append(f"Voicings: {v['n_voicings']} erkannt, "
                   f"{round(v['rootless_ratio']*100)}% rootless, "
                   f"Leitton-Abdeckung {round(v['guide_tone_coverage']*100)}%, "
                   f"im Schnitt {v['avg_tensions_per_voicing']} Tensions pro Voicing.")
    vl = r["voice_leading"]
    if vl.get("top_voice_avg_leap_semitones") is not None:
        out.append(f"Stimmfuehrung Oberstimme: {vl['smoothness_comment']} "
                   f"(Ø {vl['top_voice_avg_leap_semitones']} Halbtoene, "
                   f"{round(vl['top_voice_stepwise_ratio']*100)}% schrittweise).")
    tf = r["time_feel"]
    if tf.get("swing_ratio"):
        out.append(f"Time-Feel: {tf['feel_comment']}, {tf['timing_comment']}.")
    return out


# --- LLM-Hook: strukturierte Analyse -> didaktisches Prosa-Feedback --------

FEEDBACK_SYSTEM = (
    "Du bist ein erfahrener Jazz-Pianist und Lehrer. Du erhaeltst eine "
    "strukturierte, automatisch erzeugte Analyse eines Solo-Klavier-Chorus "
    "ueber bekannte Changes. Die Analyse stammt aus einer Transkription, die "
    "fehlerbehaftet sein kann — bewerte deshalb Tendenzen und Statistiken, "
    "nicht einzelne Noten. Gib konkretes, ermutigendes, musikalisch fundiertes "
    "Feedback in drei Teilen: (1) Was gut funktioniert, (2) zwei, drei konkrete "
    "Ansatzpunkte, (3) eine Uebe-Empfehlung. Nutze Jazz-Terminologie praezise."
)


def build_feedback_prompt(report: dict) -> dict:
    """Baut die Nachricht fuer einen LLM-Aufruf (z.B. die Anthropic-API)."""
    user = (
        "Hier die strukturierte Analyse eines Chorus. Formuliere das Feedback "
        "fuer eine Spielerin/einen Spieler.\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2) + "\n```"
    )
    return {"system": FEEDBACK_SYSTEM, "user": user}


def get_llm_feedback(report: dict, model: str = "claude-opus-4-8") -> str:
    """Optionaler Aufruf der Anthropic-API. Braucht das 'anthropic'-Paket und
    ANTHROPIC_API_KEY in der Umgebung. Ohne beides wird sauber abgebrochen."""
    prompt = build_feedback_prompt(report)
    try:
        import os, anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return "[Kein ANTHROPIC_API_KEY gesetzt — build_feedback_prompt() nutzen.]"
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=1200,
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    except ImportError:
        return "[Paket 'anthropic' nicht installiert — build_feedback_prompt() nutzen.]"
