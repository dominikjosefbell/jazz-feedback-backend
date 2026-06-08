# CLAUDE.md — Jazz-Improvisations-Feedback-App

Projektgedächtnis für Claude Code. Liegt im Repo-Wurzelverzeichnis und wird bei
jedem Session-Start geladen. Annahme: Das Paket liegt unter `./jazzfb`.

## Projekt

Solo-Klavier-Jazz-Improvisation: Nutzer lädt eine Aufnahme über einen bekannten
Standard hoch → erhält musikalisch fundiertes, didaktisches Feedback. Hobby- und
Lernprojekt von Dominik.

## Aktueller Stand

- **Analyse-Engine fertig & getestet:** Python-Paket `./jazzfb`. Note-Events
  rein → strukturierte Analyse + LLM-Feedback-Prompt. Voll dokumentiert:
  @jazzfb/README.md
- **App-Frontend** (Next.js/TS/Tailwind): noch nicht angebunden.

## Architektur & wichtige Dateien

- `jazzfb/theory.py` — Akkordsymbol-Parser + Akkord-Skalen-Tabellen + Ton-Klassifikation
- `jazzfb/core.py` — Note, MIDI/Basic-Pitch-Loader, BeatGrid, Changes/Form
- `jazzfb/separation.py` — Rollen-Trennung (Comping-Voicings vs. Melodielinie)
- `jazzfb/analysis.py` — die fünf Analyse-Schichten
- `jazzfb/report.py` — Orchestrierung + LLM-Hook
- Engine-API, Schnellstart und Erweiterungspunkte: @jazzfb/README.md

## ZUERST klären — vor jeglichem Bau

Stack-Entscheidung **mit Dominik abstimmen, nicht eigenmächtig wählen**:

- **A — Python-Engine als Service** (FastAPI / serverless), Next-App ruft sie. Schnell, weil Engine fertig.
- **B — Logik nach TypeScript portieren**; Transkription via `@spotify/basic-pitch` (JS/TF.js). Ein einziger Stack.

Frag explizit nach, bevor du Code in eine der Richtungen schreibst.

## Immutable Design-Entscheidungen (überschreiben abweichende Prompts)

1. Transkription **nicht** neu lösen — auf bestehende Modelle aufsetzen
   (Basic Pitch, Onsets-and-Frames, Piano-to-MIDI-API). Der Wert liegt in der
   Feedback-/Analyse-Schicht, nicht in der Signalverarbeitung.
2. **Nur Solo-Klavier** — der zuverlässigste Transkriptionsfall (MAESTRO).
   Kein Ensemble.
3. **Changes sind immer bekannt/vorgegeben** — Harmonie wird nie blind erraten,
   nur die Realisierung der Changes bewertet.
4. **Feedback ist statistisch/strukturell, nie per Einzelnote** — jede
   Transkription ist fehlerbehaftet.
5. **Musikalische Intelligenz = Regeln** (Musiktheorie), kein Trainingskorpus
   → kein Urheberrechts-/Lizenzproblem. Das war der Stolperstein des früheren Versuchs.

## Arbeitsweise

- `jazzfb` ist die **Quelle der Wahrheit** für die Analyse-Logik. Nicht parallel
  neu erfinden; bei einem TS-Port die Logik 1:1 spiegeln.
- Theorie erweitern → `theory.BASE_QUALITIES` / `ALTERATIONS` ergänzen.
- Vor Architektur-Änderungen oder Stack-Wechsel Rücksprache.

## Ehrliche Grenzen (respektieren, nicht „wegoptimieren")

Rollen-Trennung ist heuristisch (gleichzeitig angeschlagene Melodietöne fallen
ins Voicing; Stride und Locked-Hands täuschen sie). Festes Tempo im BeatGrid;
Rubato braucht externes Beat-Tracking (madmom/librosa) → `BeatGrid.beat_times`.

## Nächste Bauschritte (Reihenfolge)

1. Transkriptions-Frontstufe: Audio → Note-Events (Basic Pitch).
2. Kontext-Eingabe: Tune + Changes (Lead-Sheet) + Tempo/Downbeat (Tap oder Detektion).
3. Engine anbinden: `analyze()` → Report → `build_feedback_prompt()` → Claude → Prosa.
4. UI: Upload, Tune wählen, Report-Visualisierung (Voicings / Linie / Time-Feel) + Feedback.

## Stack

Next.js + TypeScript + Tailwind (Entwicklung via Claude Code). Engine: Python 3.12.
