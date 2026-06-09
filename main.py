# Jazz Improvisation Feedback Platform - MIDI VERSION
# With Key Selector + Rhythm Detection + Grand Staff Notation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import tempfile
import os
from typing import Dict, List, Optional
import json
# RAG-Wissensbasis ist optional (schwere Abhaengigkeit: chromadb). Faellt sie
# aus, laeuft die App trotzdem — nur der alte RAG-Kontext entfaellt.
try:
    from knowledge_loader import get_knowledge_base
except Exception as _kb_err:
    print(f"⚠️  knowledge_loader nicht verfuegbar ({_kb_err}) — RAG deaktiviert")
    def get_knowledge_base():
        raise RuntimeError("knowledge base unavailable")
from midi_analyzer import analyze_midi_file, analyze_voice_leading
import uuid
from datetime import datetime
import requests

# Neue Engine (jazzfb) + Standards-Bibliothek + Orchestrierung.
# Loest die alte blinde Harmonie-Erkennung ab: Changes sind bekannt/vorgegeben.
import jazz_service
import standards
import keymode

app = FastAPI(title="Jazz Feedback API - MIDI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apertus AI Configuration - NEW Router API
HF_TOKEN = os.environ.get("HF_TOKEN")
APERTUS_URL = "https://router.huggingface.co/v1/chat/completions"
APERTUS_MODEL = "swiss-ai/Apertus-8B-Instruct-2509:publicai"

def check_apertus():
    if HF_TOKEN:
        print(f"✅ Apertus AI configured (Token: {HF_TOKEN[:10]}...)")
        return True
    else:
        print("⚠️  HF_TOKEN not found - AI disabled")
        return False

apertus_enabled = check_apertus()

analysis_results = {}

# ============================================================================
# WEB UI WITH KEY SELECTOR + RHYTHM
# ============================================================================

from ui_template import HTML_TEMPLATE  # eingebettete Web-UI (ausgelagert)

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.get("/ai-status")
async def ai_status():
    return {"ai_enabled": apertus_enabled}

# ============================================================================
# JAZZ PATTERN ANALYSIS
# ============================================================================

def analyze_jazz_patterns(audio_features: Dict) -> Dict:
    tempo = audio_features.get("tempo", 120)
    note_density = audio_features.get("note_density", 2)
    
    if 60 <= tempo < 100: tempo_category, reference = "Ballad", "Bill Evans 'Waltz for Debby'"
    elif 100 <= tempo < 140: tempo_category, reference = "Medium Swing", "Miles Davis 'So What'"
    elif 140 <= tempo < 200: tempo_category, reference = "Up-Tempo", "Charlie Parker Bebop"
    else: tempo_category, reference = "Very Fast", "Coltrane 'Giant Steps'"
    
    if tempo < 100 and note_density < 2: artists = ["Bill Evans", "Keith Jarrett"]
    elif 140 <= tempo and note_density > 3: artists = ["Charlie Parker", "Dizzy Gillespie"]
    else: artists = ["Miles Davis", "Dexter Gordon"]
    
    return {
        "tempo_category": tempo_category,
        "tempo_reference": reference,
        "rhythm_assessment": "Gut",
        "density_assessment": "Ausgewogen",
        "swing_feel": "Swing",
        "similar_artists": artists
    }

# ============================================================================
# APERTUS AI FEEDBACK
# ============================================================================

async def get_apertus_feedback(audio_features: Dict, jazz_analysis: Dict, note_analysis: Dict, user_key: str) -> Dict:
    if not apertus_enabled or not HF_TOKEN:
        return None
    
    try:
        try:
            kb = get_knowledge_base()
            jazz_context = kb.get_context_for_analysis(tempo=audio_features.get('tempo', 120), tempo_category=jazz_analysis['tempo_category'], rhythm_complexity=5)
        except: jazz_context = ""
        
        chord_info = ""
        if note_analysis and note_analysis.get("chords"):
            # Get unique chord symbols
            symbols = []
            last = ""
            for c in note_analysis['chords']:
                if c.get('symbol', '?') != last:
                    symbols.append(c.get('symbol', '?'))
                    last = c.get('symbol', '?')
            chord_info = f"Akkorde: {' → '.join(symbols)}"
            if note_analysis.get('progression', {}).get('type'):
                chord_info += f" ({note_analysis['progression']['type']})"
        
        prompt = f"""Du bist ein Jazz-Lehrer. Analysiere diese Improvisation in {user_key}:

- Tempo: {audio_features.get('tempo', 120):.0f} BPM
- Noten: {note_analysis.get('total_notes', 0)}
- {chord_info}
- Stil: {jazz_analysis['tempo_category']}

Gib Feedback (1-10) für: Rhythmus, Harmonie, Melodie, Artikulation.
Antworte NUR als JSON:
{{"rhythm": {{"score": 7.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "harmony": {{"score": 8.0, "feedback": "...", "tips": ["...", "...", "..."]}}, "melody": {{"score": 6.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "articulation": {{"score": 7.0, "feedback": "...", "tips": ["...", "...", "..."]}}}}"""
        
        # NEW: Use Router API with requests
        response = requests.post(
            APERTUS_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": APERTUS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1200,
                "temperature": 0.7
            },
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"Apertus API Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        text = result['choices'][0]['message']['content']
        text = text.replace('```json', '').replace('```', '').strip()
        if "{" in text: text = text[text.find("{"):text.rfind("}")+1]
        return json.loads(text)
    except Exception as e:
        print(f"Apertus Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_rule_based_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    return {
        "rhythm": {"score": 7.0, "feedback": "Solides Timing.", "tips": ["Metronom üben", "Swing-Feel entwickeln", "Synkopen einbauen"]},
        "harmony": {"score": 7.0, "feedback": "Gute Akkordwahl.", "tips": ["Guide Tones betonen", "Chromatik nutzen", "Voice Leading verbessern"]},
        "melody": {"score": 6.5, "feedback": "Interessante Melodien.", "tips": ["Phrasenlängen variieren", "Pausen nutzen", "Motivische Entwicklung"]},
        "articulation": {"score": 7.0, "feedback": "Gute Dynamik.", "tips": ["Kontraste verstärken", "Akzente setzen", "Legato/Staccato mischen"]}
    }

# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_midi_in_background(analysis_id: str, tmp_path: str, user_key: str):
    import asyncio, gc
    try:
        analysis_results[analysis_id] = {"status": "processing", "stage": "notes"}
        
        note_analysis = analyze_midi_file(tmp_path)
        note_analysis['detected_scale'] = user_key
        
        if note_analysis.get('error') or note_analysis.get('total_notes', 0) == 0:
            raise Exception(f"MIDI failed: {note_analysis.get('error', 'No notes')}")
        
        from midi_analyzer import detect_progression
        note_analysis['progression'] = detect_progression(note_analysis.get('chords', []), user_key)
        
        duration = max(1, note_analysis.get('duration', 1))
        audio_features = {
            "duration": duration,
            "tempo": note_analysis.get('tempo_bpm', 120),
            "tempo_stability": note_analysis.get('timing', {}).get('precision_score', 0.8),
            "note_density": note_analysis.get('total_notes', 0) / duration,
            "dynamics": {"dynamic_range": note_analysis.get('dynamics', {}).get('range', 40) / 127},
            "rhythm_complexity": 5,
        }
        
        jazz_analysis = analyze_jazz_patterns(audio_features)
        gc.collect()
        
        analysis_results[analysis_id] = {"status": "processing", "stage": "ai"}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        feedback = loop.run_until_complete(get_apertus_feedback(audio_features, jazz_analysis, note_analysis, user_key))
        loop.close()
        
        if not feedback: feedback = generate_rule_based_feedback(audio_features, jazz_analysis)
        
        overall_score = (feedback["rhythm"]["score"] + feedback["harmony"]["score"] + feedback["melody"]["score"] + feedback["articulation"]["score"]) / 4
        
        analysis_results[analysis_id] = {"status": "completed", "result": {
            "overall_score": round(overall_score, 1),
            "audio_features": audio_features,
            "jazz_analysis": jazz_analysis,
            "note_analysis": note_analysis,
            "feedback": feedback,
            "ai_generated": apertus_enabled,
            "user_key": user_key
        }}
    except Exception as e:
        import traceback; traceback.print_exc()
        analysis_results[analysis_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

# ============================================================================
# NEUE ENGINE: jazzfb gegen BEKANNTE Changes (Slice 1)
# ============================================================================

async def get_apertus_feedback_grounded(facts: str, context_label: str) -> Optional[Dict]:
    """Apertus-Feedback, das auf den regelbasierten jazzfb-Fakten fusst.
    Behaelt das bestehende Score-JSON-Format (rhythm/harmony/melody/
    articulation), damit die UI unveraendert rendert. Die Fakten enthalten
    bereits den Harmonie-Kontext (volle Changes / Tonart / keiner) — der
    Prompt ist deshalb kontext-neutral gehalten."""
    if not apertus_enabled or not HF_TOKEN:
        return None
    try:
        prompt = f"""Du bist ein erfahrener Jazz-Pianist und Klavier-Lehrer. Unten steht eine
AUTOMATISCH ERZEUGTE, REGELBASIERTE Analyse eines Solo-Klavier-Stuecks ({context_label}).
Die erste Zeile der Fakten nennt den Harmonie-Kontext (volle Changes, eine
Tonart/Tonleiter, oder KEINER). Wenn kein bzw. nur Tonart-Kontext vorliegt, mach
KEINE Aussagen ueber einzelne Akkorde — bewerte Harmonie dann zurueckhaltend bzw.
nur auf Tonart-Ebene. Die Zahlen stammen aus Musiktheorie-Regeln und aus einer
moeglicherweise fehlerbehafteten Transkription — bewerte TENDENZEN und STATISTIK,
nicht einzelne Noten.

ANALYSE-FAKTEN:
{facts}

Gib didaktisches, ermutigendes, musikalisch fundiertes Feedback. Bewerte (1.0-10.0)
und begruende kurz fuer: Rhythmus/Time-Feel, Harmonie (so weit der Kontext es
hergibt), Melodie/Linienfuehrung, Artikulation/Dynamik. Jede Kategorie mit
3 konkreten Uebe-Tipps.
Antworte NUR als JSON:
{{"rhythm": {{"score": 7.5, "feedback": "...", "tips": ["...","...","..."]}}, "harmony": {{"score": 8.0, "feedback": "...", "tips": ["...","...","..."]}}, "melody": {{"score": 6.5, "feedback": "...", "tips": ["...","...","..."]}}, "articulation": {{"score": 7.0, "feedback": "...", "tips": ["...","...","..."]}}}}"""

        response = requests.post(
            APERTUS_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json={"model": APERTUS_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 1400, "temperature": 0.7},
            timeout=90,
        )
        if response.status_code != 200:
            print(f"Apertus API Error: {response.status_code} - {response.text}")
            return None
        text = response.json()['choices'][0]['message']['content']
        text = text.replace('```json', '').replace('```', '').strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}") + 1]
        return json.loads(text)
    except Exception as e:
        print(f"Apertus (grounded) Error: {e}")
        import traceback; traceback.print_exc()
        return None


def _finish_jazz_analysis(analysis_id: str, res: dict):
    """Gemeinsamer Abschluss fuer MIDI- und Audio/Note-Events-Pfad:
    Fakten -> Apertus -> Score -> Ergebnis ablegen."""
    import asyncio
    if not res.get("ok"):
        analysis_results[analysis_id] = {"status": "error",
                                         "error": res.get("error", "Analyse fehlgeschlagen.")}
        return
    report, summary, used = res["report"], res["summary"], res["used"]
    label = report.get("context", {}).get("label", "Ohne Harmonie-Kontext")
    facts = jazz_service.facts_for_llm(report, summary, label)

    analysis_results[analysis_id] = {"status": "processing", "stage": "ai"}
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    feedback = loop.run_until_complete(get_apertus_feedback_grounded(facts, label))
    loop.close()

    ai_generated = feedback is not None
    if not feedback:
        feedback = generate_rule_based_feedback({}, {})
    overall = (feedback["rhythm"]["score"] + feedback["harmony"]["score"]
               + feedback["melody"]["score"] + feedback["articulation"]["score"]) / 4

    analysis_results[analysis_id] = {"status": "completed", "result": {
        "overall_score": round(overall, 1),
        "tune": label,
        "report": report,
        "summary": summary,
        "used": used,
        "facts": facts,
        "feedback": feedback,
        "ai_generated": ai_generated,
    }}


def process_midi_jazz(analysis_id: str, tmp_path: str, context: dict,
                      beats_per_bar: int, bpm: Optional[float]):
    import gc
    try:
        analysis_results[analysis_id] = {"status": "processing", "stage": "notes"}
        res = jazz_service.analyze_midi(
            tmp_path, context,
            beats_per_bar=(int(beats_per_bar) if beats_per_bar else None),
            bpm=(float(bpm) if bpm else None))
        gc.collect()
        _finish_jazz_analysis(analysis_id, res)
    except Exception as e:
        import traceback; traceback.print_exc()
        analysis_results[analysis_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def process_notes_jazz(analysis_id: str, note_events: list, context: dict,
                       beats_per_bar: int, bpm: Optional[float]):
    """Pfad fuer im Browser transkribierte Audio-Aufnahmen (Basic Pitch)."""
    import gc
    try:
        analysis_results[analysis_id] = {"status": "processing", "stage": "notes"}
        res = jazz_service.analyze_notes(
            note_events, context,
            beats_per_bar=(int(beats_per_bar) if beats_per_bar else None),
            bpm=(float(bpm) if bpm else None))
        gc.collect()
        _finish_jazz_analysis(analysis_id, res)
    except Exception as e:
        import traceback; traceback.print_exc()
        analysis_results[analysis_id] = {"status": "error", "error": str(e)}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/standards")
async def get_standards():
    """Liste der bekannten Standards (Anzeigenamen) fuer die UI."""
    return {"standards": standards.list_standards()}


@app.get("/modes")
async def get_modes():
    """Liste der Tonart-Modi (value/label) fuer die UI."""
    return {"modes": keymode.list_modes()}


@app.post("/analyze-jazz")
async def analyze_jazz(background_tasks: BackgroundTasks,
                       file: UploadFile = File(...),
                       tune: str = Form(""),
                       manual_changes: str = Form(""),
                       key_tonic: str = Form(""),
                       key_mode: str = Form(""),
                       beats_per_bar: int = Form(4),
                       bpm: str = Form("")):
    """MIDI + OPTIONALER Harmonie-Kontext -> jazzfb-Analyse -> Apertus.
    Kontext-Precedence: eigene Changes > Tune > Tonart > keiner."""
    if not (file.filename.endswith('.mid') or file.filename.endswith('.midi')):
        raise HTTPException(status_code=400, detail="Aktuell nur MIDI-Dateien (.mid/.midi)")
    analysis_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        bpm_val = float(bpm) if bpm.strip() else None
    except ValueError:
        bpm_val = None
    context = jazz_service.resolve_context(tune or None, manual_changes or None,
                                           key_tonic or None, key_mode or None)
    background_tasks.add_task(process_midi_jazz, analysis_id, tmp_path,
                             context, beats_per_bar, bpm_val)
    return {"analysis_id": analysis_id, "status": "processing"}


@app.post("/analyze-notes")
async def analyze_notes_endpoint(background_tasks: BackgroundTasks, request: Request):
    """Audio-Pfad: der Browser transkribiert mit Basic Pitch und schickt die
    Note-Events als JSON. Body:
      { notes: [[start_s, end_s, pitch_midi, amplitude], ...],
        tune?, manual_changes?, key_tonic?, key_mode?, beats_per_bar?, bpm? }"""
    body = await request.json()
    note_events = body.get("notes") or []
    if not note_events:
        raise HTTPException(status_code=400, detail="Keine Note-Events uebergeben.")
    try:
        bpm_val = float(body["bpm"]) if str(body.get("bpm", "")).strip() else None
    except (ValueError, TypeError):
        bpm_val = None
    beats_per_bar = int(body.get("beats_per_bar") or 4)
    context = jazz_service.resolve_context(
        body.get("tune") or None, body.get("manual_changes") or None,
        body.get("key_tonic") or None, body.get("key_mode") or None)
    analysis_id = str(uuid.uuid4())
    background_tasks.add_task(process_notes_jazz, analysis_id, note_events,
                             context, beats_per_bar, bpm_val)
    return {"analysis_id": analysis_id, "status": "processing"}


@app.post("/analyze-async")
async def analyze_midi_async(background_tasks: BackgroundTasks, file: UploadFile = File(...), key: str = Form("C Major")):
    if not (file.filename.endswith('.mid') or file.filename.endswith('.midi')):
        raise HTTPException(status_code=400, detail="Nur MIDI-Dateien erlaubt")
    
    analysis_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    background_tasks.add_task(process_midi_in_background, analysis_id, tmp_path, key)
    return {"analysis_id": analysis_id, "status": "processing"}

@app.get("/result/{analysis_id}")
async def get_result(analysis_id: str):
    if analysis_id not in analysis_results: raise HTTPException(status_code=404, detail="Not found")
    return analysis_results[analysis_id]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_enabled": apertus_enabled}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)