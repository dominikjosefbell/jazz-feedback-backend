"""
example.py — End-to-End-Demo ohne externe Abhaengigkeiten.

Baut synthetische Note-Events (ii-V-I in C: Dm7 | G7 | Cmaj7 | Cmaj7) mit
rootless Comping-Voicings und einer geswungten rechten Hand und laesst die
gesamte Analyse-Logik darueberlaufen. Im echten Einsatz ersetzt du die
synthetischen Noten durch from_midi(...) oder from_basic_pitch(...).
"""

import json
from jazzfb import Note, BeatGrid, Changes, analyze, rule_based_summary, build_feedback_prompt

BPM = 120
SPB = 60 / BPM    # 0.5 s pro Beat


def n(onset, pitch, dur=0.45, vel=80):
    return Note(onset, onset + dur, pitch, vel)


# --- Comping: rootless Voicings (linke Hand) -------------------------------
clusters = [
    # Dm7 (3-5-7-9 = F A C E)
    [n(0.0, p, 1.8, 62) for p in (65, 69, 72, 76)],
    # G7 (3-13-7-9 = B E F A)
    [n(2.0, p, 1.8, 62) for p in (59, 64, 65, 69)],
    # Cmaj7 (3-5-7-9 = E G B D)
    [n(4.0, p, 1.8, 62) for p in (64, 67, 71, 74)],
    [n(6.0, p, 1.8, 62) for p in (64, 67, 71, 74)],
]

# --- Linie: geswungte rechte Hand ------------------------------------------
# Swing: erste Achtel ~0.33s, zweite ~0.17s (ca. 2:1)
line_spec = [
    # Bar 0 Dm7
    (0.00, 62), (0.33, 64), (0.50, 65), (0.83, 67),
    (1.00, 69), (1.33, 71), (1.50, 72), (1.83, 71),
    # Bar 1 G7  (C5 auf Beat 3 = Avoid-Note auf betonter Zeit -> Test)
    (2.00, 74), (2.33, 76), (2.50, 77), (2.83, 76),
    (3.00, 72), (3.33, 71), (3.50, 69), (3.83, 67),
    # Bar 2 Cmaj7
    (4.00, 76), (4.33, 74), (4.50, 72), (4.83, 71),
    (5.00, 67), (5.33, 69), (5.50, 67),
    # Bar 3 Cmaj7
    (6.00, 76), (6.50, 72),
]
line_notes = []
for i, (onset, pitch) in enumerate(line_spec):
    nxt = line_spec[i + 1][0] if i + 1 < len(line_spec) else onset + 0.45
    line_notes.append(Note(onset, onset + min(0.45, (nxt - onset) * 0.9), pitch, 78))

notes = [nt for grp in clusters for nt in grp] + line_notes

grid = BeatGrid(bpm=BPM, start=0.0, beats_per_bar=4)
changes = Changes.from_bars([["Dm7"], ["G7"], ["Cmaj7"], ["Cmaj7"]])

report = analyze(notes, grid, changes)

print("=" * 70)
print("REGELBASIERTE ZUSAMMENFASSUNG (ohne LLM):")
print("=" * 70)
for line in rule_based_summary(report):
    print(" • " + line)

print("\n" + "=" * 70)
print("STRUKTURIERTE ANALYSE (Auszug):")
print("=" * 70)
compact = {k: report[k] for k in ("meta", "voicings", "voice_leading", "time_feel", "contour")}
compact["line"] = {k: report["line"][k] for k in
                   ("n_notes", "distribution", "chord_tones_on_strong_beats",
                    "avoid_notes_on_strong_beats")}
print(json.dumps(compact, ensure_ascii=False, indent=2))

print("\n" + "=" * 70)
print("LLM-PROMPT (geht so an die Anthropic-API fuer Prosa-Feedback):")
print("=" * 70)
p = build_feedback_prompt(report)
print("SYSTEM:", p["system"][:160], "...\n")
print("USER (Anfang):", p["user"][:240], "...")
