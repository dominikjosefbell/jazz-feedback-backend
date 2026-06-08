# jazzfb — Solo-Klavier Jazz-Improvisations-Analyse

Regelbasierte, **modellunabhängige** Engine: Sie bekommt **Note-Events**
(Onset, Offset, Tonhöhe, Velocity) und erzeugt eine strukturierte, musikalisch
fundierte Analyse eines Chorus über **bekannte Changes**. Die Transkription
(Audio → Noten) liefert ein externes Modell — die Engine ist davon entkoppelt.

Kein Training, kein Korpus, kein Lizenzproblem: Die musikalische Intelligenz
steckt in Musiktheorie-Regeln, nicht in trainierten Gewichten.

## Architektur (Signalkette)

```
Audio ──[externes Transkriptionsmodell]──> Note-Events
                                              │
        from_midi() / from_basic_pitch()  ───┤
                                              ▼
   Beat-Raster (BeatGrid) + Changes (Form) ──> separate()  (Comping vs. Linie)
                                              ▼
                                         analyze()
            ┌───────────────┬───────────────┬───────────────┬───────────┐
         Voicings        Linie         Voice-Leading     Time-Feel    Kontur
       (rootless,    (Akkordton-/      (Leittonlinien,   (Swing-Ratio, (Range,
        Tensions,     Tension-Stats,    Glätte)          Lage z. Beat) Dynamik)
        Leittöne)     Avoid-Noten)
                                              ▼
                       strukturierter Report (JSON)
                          │                       │
                rule_based_summary()      build_feedback_prompt() ──> Claude
                  (ohne LLM)                                          (Prosa-Feedback)
```

## Module

- `theory.py` — Akkordsymbol-Parser + Akkord-Skalen-Tabellen + Ton-Klassifikation
- `core.py` — Note, MIDI/Basic-Pitch-Loader, BeatGrid, Changes/Form
- `separation.py` — Hand-/Rollen-Trennung (Onset-Gleichzeitigkeit)
- `analysis.py` — die fünf analytischen Schichten
- `report.py` — Orchestrierung, Regel-Zusammenfassung, LLM-Hook
- `example.py` — lauffähige Demo (synthetische ii–V–I, keine Abhängigkeiten)

## Schnellstart

```bash
python3 -m jazzfb.example     # Demo, läuft ohne Installation
```

Echter Einsatz mit einer MIDI-Datei oder Basic-Pitch-Ausgabe:

```python
from jazzfb import from_midi, BeatGrid, Changes, analyze, build_feedback_prompt

notes   = from_midi("mein_solo.mid")                 # oder from_basic_pitch(...)
grid    = BeatGrid(bpm=140, start=0.0, beats_per_bar=4)
changes = Changes.from_bars([["Dm7"], ["G7"], ["Cmaj7"], ["Cmaj7"]])

report  = analyze(notes, grid, changes)
prompt  = build_feedback_prompt(report)              # -> an die Anthropic-API
```

## Transkription anbinden

- **Spotify Basic Pitch** (`pip install basic-pitch`): liefert `note_events`
  → `from_basic_pitch(note_events)`. Leichtgewichtig, guter Start.
- **Magenta Onsets-and-Frames / MT3**: nach MIDI exportieren → `from_midi()`.
- **Piano-to-MIDI-API** (z. B. Klangio): MIDI zurück → `from_midi()`.

Solo-Klavier ist der zuverlässigste Transkriptionsfall (MAESTRO-Datensatz) —
deshalb die bewusste Beschränkung auf Solo.

## Bewusste Grenzen (ehrlich)

- **Rollen-Trennung ist heuristisch.** Gleichzeitig mit einem Voicing
  angeschlagene Melodietöne werden dem Cluster zugeschlagen (im Demo landet
  die Melodie-D auf dem Downbeat im Dm7-Voicing). Stride und Locked-Hands
  täuschen die Heuristik. Darum bleibt die Auswertung **statistisch**, nie
  per Einzelnote.
- **Changes müssen bekannt sein.** Die Engine vergleicht gegen eine Referenz-
  Form; sie erkennt Harmonie nicht aus dem Nichts.
- **Festes Tempo** im BeatGrid. Für rubato/variables Tempo extern Beats
  tracken (madmom/librosa) und `BeatGrid.beat_times` füttern.

## Erweiterungspunkte

- Akkord-Vokabular in `theory.BASE_QUALITIES` / `ALTERATIONS` ergänzen.
- Voice-Streaming in `separation.py` durch einen echten Viterbi-Ansatz ersetzen.
- Chromatische Approach-Noten in `analysis.analyze_line` mit Lookahead als
  „korrekte Annäherung" statt „chromatisch" labeln.
- Optionales ML erst später (Voice-Separation verbessern, Stilklassifikation) —
  gegen die Baseline antreten lassen.

## Optionale Abhängigkeiten

```bash
pip install pretty_midi    # MIDI-Laden (empfohlen)
pip install basic-pitch    # Audio -> Note-Events
pip install anthropic      # LLM-Feedback
```
