# Jazz Improvisation Feedback Platform - MIDI VERSION WITH PIANO SHEET MUSIC
# FastAPI + MIDI Analysis + Apertus AI + Grand Staff Notation

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
import tempfile
import os
from typing import Dict, List, Optional
import json
from knowledge_loader import get_knowledge_base
from midi_analyzer import analyze_midi_file, analyze_voice_leading
import uuid
from datetime import datetime
from huggingface_hub import InferenceClient

app = FastAPI(title="Jazz Feedback API - MIDI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apertus API Client
apertus_client = None

def initialize_apertus():
    """Initialize Apertus via Hugging Face"""
    global apertus_client
    try:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            apertus_client = InferenceClient(token=hf_token)
            print("✅ Apertus API connected!")
        else:
            print("⚠️  HF_TOKEN not found - AI disabled")
    except Exception as e:
        print(f"⚠️  Apertus init failed: {e}")

initialize_apertus()

# In-Memory Storage
analysis_results = {}

# ============================================================================
# WEB UI WITH GRAND STAFF NOTATION
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback - MIDI Analyse</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- ABCJS for Sheet Music Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/abcjs@6.2.2/dist/abcjs-basic-min.js"></script>
    <style>
        .abcjs-container svg { max-width: 100%; height: auto; }
        #pianoSheet svg { background: white; border-radius: 8px; }
    </style>
</head>
<body class="bg-gradient-to-br from-red-50 via-white to-white min-h-screen">
    <div class="max-w-4xl mx-auto p-6">
        <div class="text-center mb-8 pt-8">
            <div class="inline-flex items-center justify-center w-16 h-16 bg-red-600 rounded-full mb-4">
                <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
                </svg>
            </div>
            <h1 class="text-4xl font-bold text-gray-900 mb-2">Jazz-Improvisation Feedback</h1>
            <p class="text-gray-600">🇨🇭 Apertus AI + 🎹 MIDI Analyse + 🎼 Notenschrift</p>
        </div>

        <div id="aiStatus" class="mb-6"></div>

        <div class="bg-white rounded-2xl shadow-lg p-8 mb-6">
            <input type="file" id="fileInput" accept=".mid,.midi" class="hidden">
            <div id="dropzone" class="border-3 border-dashed border-red-300 rounded-xl p-12 text-center cursor-pointer hover:border-red-500 hover:bg-red-50 transition-all">
                <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg font-semibold text-gray-700 mb-2" id="fileName">MIDI-Datei hochladen</p>
                <p class="text-sm text-gray-500">MIDI-Dateien (.mid, .midi)</p>
            </div>

            <button id="analyzeBtn" class="w-full mt-6 bg-red-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-red-700 disabled:bg-gray-400 transition-colors hidden">
                🎵 MIDI Analyse starten
            </button>
        </div>

        <div id="loading" class="bg-white rounded-2xl shadow-lg p-6 mb-6 hidden">
            <div class="flex items-center gap-3 mb-3">
                <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600"></div>
                <span class="text-gray-700 font-medium" id="loadingText">Analysiere...</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
                <div id="progressBar" class="bg-red-600 h-2 rounded-full transition-all" style="width: 0%"></div>
            </div>
        </div>

        <div id="results" class="space-y-6"></div>
    </div>

    <script>
        let selectedFile = null;
        let analysisId = null;

        fetch('/ai-status').then(r => r.json()).then(data => {
            const statusDiv = document.getElementById('aiStatus');
            if (data.ai_enabled) {
                statusDiv.innerHTML = '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="flex items-center gap-3"><svg class="w-8 h-8 text-red-600" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg><div><div class="text-sm font-medium text-red-900">🇨🇭 Apertus AI + 🎹 MIDI + 🎼 Notenschrift</div><div class="text-xs text-red-700">Akkord-Erkennung & Swiss AI Feedback</div></div></div></div>';
            } else {
                statusDiv.innerHTML = '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">⚠️ AI deaktiviert</span></div>';
            }
        });

        document.getElementById('dropzone').addEventListener('click', () => document.getElementById('fileInput').click());

        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && (file.name.endsWith('.mid') || file.name.endsWith('.midi'))) {
                selectedFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('analyzeBtn').classList.remove('hidden');
            } else {
                alert('Bitte eine MIDI-Datei (.mid oder .midi) auswählen');
            }
        });

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            document.getElementById('loadingText').textContent = 'Uploading...';
            document.getElementById('progressBar').style.width = '10%';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/analyze-async', { method: 'POST', body: formData });
                const data = await response.json();
                analysisId = data.analysis_id;
                document.getElementById('loadingText').textContent = 'Analysiere MIDI...';
                document.getElementById('progressBar').style.width = '20%';
                pollForResults(analysisId);
            } catch (error) {
                alert('Fehler: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
            }
        });

        async function pollForResults(id) {
            const maxAttempts = 60;
            let attempts = 0;
            const interval = setInterval(async () => {
                attempts++;
                try {
                    const response = await fetch('/result/' + id);
                    const data = await response.json();
                    if (data.status === 'completed') {
                        clearInterval(interval);
                        document.getElementById('progressBar').style.width = '100%';
                        setTimeout(() => {
                            document.getElementById('loading').classList.add('hidden');
                            displayResults(data.result);
                        }, 500);
                    } else if (data.status === 'processing') {
                        const progress = 20 + (attempts / maxAttempts) * 70;
                        document.getElementById('progressBar').style.width = progress + '%';
                        if (data.stage === 'notes') document.getElementById('loadingText').textContent = '🎹 MIDI Analyse...';
                        else if (data.stage === 'ai') document.getElementById('loadingText').textContent = '🇨🇭 Apertus AI Feedback...';
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        alert('Fehler: ' + data.error);
                        document.getElementById('loading').classList.add('hidden');
                    }
                    if (attempts >= maxAttempts) {
                        clearInterval(interval);
                        alert('Timeout');
                        document.getElementById('loading').classList.add('hidden');
                    }
                } catch (error) { console.error('Poll error:', error); }
            }, 3000);
        }

        // ABC Notation Helper - MIDI to ABC for Treble Clef
        function midiToAbcTreble(midiNote) {
            const noteNames = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B'];
            const octave = Math.floor(midiNote / 12) - 1;
            const note = noteNames[midiNote % 12];
            if (octave === 4) return note;
            if (octave === 5) return note.toLowerCase();
            if (octave === 6) return note.toLowerCase() + "'";
            if (octave === 7) return note.toLowerCase() + "''";
            if (octave === 3) return note + ",";
            if (octave === 2) return note + ",,";
            return note;
        }

        // ABC Notation Helper - MIDI to ABC for Bass Clef
        function midiToAbcBass(midiNote) {
            const noteNames = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B'];
            const octave = Math.floor(midiNote / 12) - 1;
            const note = noteNames[midiNote % 12];
            if (octave === 2) return note + ",";
            if (octave === 3) return note;
            if (octave === 4) return note.toLowerCase();
            if (octave === 1) return note + ",,";
            return note + ",";
        }

        // Generate Grand Staff (Piano) ABC Notation
        function generateGrandStaffAbc(notes, chords, key, tempo) {
            const middleC = 60;
            const bpm = tempo || 120;
            const beatDuration = 60 / bpm; // seconds per beat
            
            // Group notes by beat
            const timeGroups = {};
            notes.forEach(note => {
                const beat = Math.floor(note.start_time / beatDuration);
                if (!timeGroups[beat]) timeGroups[beat] = { treble: [], bass: [] };
                if (note.pitch >= middleC) {
                    timeGroups[beat].treble.push(note.pitch);
                } else {
                    timeGroups[beat].bass.push(note.pitch);
                }
            });

            // Chord symbols per beat
            const chordAtBeat = {};
            if (chords) {
                chords.forEach(chord => {
                    const beat = Math.floor(chord.start_time / beatDuration);
                    chordAtBeat[beat] = chord.symbol;
                });
            }

            const beats = Object.keys(timeGroups).map(Number).sort((a, b) => a - b);
            const maxBeats = Math.min(beats.length, 32);
            
            // Extract key for ABC
            let abcKey = 'C';
            if (key) {
                const keyMatch = key.match(/^([A-G][#b]?)/);
                if (keyMatch) abcKey = keyMatch[1].replace('#', '^').replace('b', '_');
            }

            let abc = "X:1\\n";
            abc += "T:MIDI Analyse\\n";
            abc += "M:4/4\\n";
            abc += "L:1/4\\n";
            abc += "Q:1/4=" + bpm + "\\n";
            abc += "K:" + abcKey + "\\n";
            abc += "%%staves {1 2}\\n";
            abc += "V:1 clef=treble\\n";
            abc += "V:2 clef=bass\\n";
            
            // Treble staff
            abc += "[V:1] ";
            let count = 0;
            for (let i = 0; i < maxBeats; i++) {
                const beat = beats[i];
                if (beat === undefined) continue;
                const group = timeGroups[beat];
                
                if (chordAtBeat[beat]) abc += '"' + chordAtBeat[beat] + '"';
                
                if (!group.treble || group.treble.length === 0) {
                    abc += "z";
                } else if (group.treble.length === 1) {
                    abc += midiToAbcTreble(group.treble[0]);
                } else {
                    abc += "[";
                    group.treble.sort((a,b) => a-b).forEach(p => abc += midiToAbcTreble(p));
                    abc += "]";
                }
                count++;
                if (count % 4 === 0) abc += " |";
                abc += " ";
            }
            abc += "|\\n";
            
            // Bass staff
            abc += "[V:2] ";
            count = 0;
            for (let i = 0; i < maxBeats; i++) {
                const beat = beats[i];
                if (beat === undefined) continue;
                const group = timeGroups[beat];
                
                if (!group.bass || group.bass.length === 0) {
                    abc += "z";
                } else if (group.bass.length === 1) {
                    abc += midiToAbcBass(group.bass[0]);
                } else {
                    abc += "[";
                    group.bass.sort((a,b) => a-b).forEach(p => abc += midiToAbcBass(p));
                    abc += "]";
                }
                count++;
                if (count % 4 === 0) abc += " |";
                abc += " ";
            }
            abc += "|";
            
            return abc;
        }

        function displayResults(data) {
            const getScoreColor = (score) => score >= 8 ? 'green' : score >= 6 ? 'yellow' : 'orange';
            let html = '';

            if (data.ai_generated) {
                html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-4 mb-6"><div class="flex items-center gap-2"><span class="font-semibold">🇨🇭 Feedback von Apertus AI</span></div></div>';
            }

            // GRAND STAFF SHEET MUSIC (Single Piano Display)
            if (data.note_analysis && data.note_analysis.notes && data.note_analysis.notes.length > 0) {
                html += '<div class="bg-white border-2 border-gray-200 rounded-2xl p-6 mb-4 shadow-lg">';
                html += '<h3 class="font-bold text-xl text-gray-900 mb-2">🎼 Notenschrift</h3>';
                html += '<p class="text-sm text-gray-500 mb-4">Klaviersystem mit Violin- und Bassschlüssel</p>';
                html += '<div id="pianoSheet" class="bg-gray-50 rounded-xl p-4 overflow-x-auto min-h-[200px]"></div>';
                html += '</div>';
            }

            // NOTE DETECTION
            if (data.note_analysis && data.note_analysis.total_notes > 0) {
                html += '<div class="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6 mb-4">';
                html += '<h3 class="font-semibold text-purple-900 mb-4">🎹 Erkannte Noten</h3>';
                html += '<div class="grid grid-cols-2 gap-4 text-sm">';
                html += '<div><strong>Anzahl:</strong> ' + data.note_analysis.total_notes + '</div>';
                html += '<div><strong>Tonumfang:</strong> ' + data.note_analysis.pitch_range.min_note + ' - ' + data.note_analysis.pitch_range.max_note + '</div>';
                html += '<div><strong>Häufigste:</strong> ' + data.note_analysis.most_common_notes.join(', ') + '</div>';
                html += '<div><strong>Tonart:</strong> ' + (data.note_analysis.detected_scale || 'unbekannt') + '</div>';
                html += '</div></div>';
            }

            // CHORD DETECTION
            if (data.note_analysis && data.note_analysis.chords && data.note_analysis.chords.length > 0) {
                html += '<div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6 mb-4">';
                html += '<h3 class="font-semibold text-blue-900 mb-4">🎼 Erkannte Akkorde</h3>';
                html += '<div class="flex flex-wrap gap-3 mb-4 items-center">';
                data.note_analysis.chords.forEach((chord, index) => {
                    const isMinor = chord.type && chord.type.includes('m') && !chord.type.includes('maj');
                    const bgColor = isMinor ? 'bg-indigo-100 border-indigo-300' : 'bg-blue-100 border-blue-300';
                    html += '<div class="' + bgColor + ' border-2 rounded-xl px-4 py-3 text-center shadow-sm">';
                    html += '<div class="font-bold text-xl text-gray-800">' + chord.symbol + '</div>';
                    html += '<div class="text-sm text-gray-600 mt-1">' + chord.notes.join(' · ') + '</div>';
                    html += '<div class="text-xs text-gray-400 mt-1">' + chord.start_time.toFixed(2) + 's</div>';
                    html += '</div>';
                    if (index < data.note_analysis.chords.length - 1) html += '<div class="text-gray-400 text-2xl">→</div>';
                });
                html += '</div>';
                if (data.note_analysis.progression) {
                    html += '<div class="bg-white/60 rounded-lg p-4 mt-3">';
                    html += '<p class="font-semibold text-blue-800">' + data.note_analysis.progression.analysis + '</p>';
                    if (data.note_analysis.progression.roman_numerals && data.note_analysis.progression.roman_numerals.length > 0) {
                        html += '<p class="text-sm text-blue-600 mt-2"><strong>Roman Numerals:</strong> ' + data.note_analysis.progression.roman_numerals.join(' → ') + '</p>';
                    }
                    if (data.note_analysis.progression.type && data.note_analysis.progression.type !== 'Custom') {
                        html += '<p class="text-sm text-green-600 mt-1">✓ Erkannt als: <strong>' + data.note_analysis.progression.type + '</strong></p>';
                    }
                    html += '</div>';
                }
                html += '</div>';
            }

            // JAZZ CONTEXT
            html += '<div class="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6 mb-4"><h3 class="font-semibold text-red-900 mb-4">🎷 Jazz-Kontext</h3><div class="space-y-2 text-sm"><p><strong>Tempo:</strong> ' + data.jazz_analysis.tempo_category + '</p><p class="text-red-700">' + data.jazz_analysis.tempo_reference + '</p><p><strong>Rhythmik:</strong> ' + data.jazz_analysis.rhythm_assessment + '</p><p><strong>Dichte:</strong> ' + data.jazz_analysis.density_assessment + '</p><p><strong>Swing:</strong> ' + data.jazz_analysis.swing_feel + '</p><p><strong>Ähnlich:</strong> ' + data.jazz_analysis.similar_artists.join(', ') + '</p></div></div>';

            // MESSWERTE
            html += '<div class="bg-slate-50 border border-slate-200 rounded-xl p-6 mb-4"><h3 class="font-semibold text-slate-900 mb-4">📊 Messwerte</h3><div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm"><div><p class="text-slate-600">Dauer</p><p class="font-mono font-bold">' + data.audio_features.duration.toFixed(1) + 's</p></div><div><p class="text-slate-600">Tempo</p><p class="font-mono font-bold">' + data.audio_features.tempo.toFixed(1) + ' BPM</p></div><div><p class="text-slate-600">Stabilität</p><p class="font-mono font-bold">' + (data.audio_features.tempo_stability * 100).toFixed(0) + '%</p></div><div><p class="text-slate-600">Noten-Dichte</p><p class="font-mono font-bold">' + data.audio_features.note_density.toFixed(2) + '/s</p></div><div><p class="text-slate-600">Dynamik</p><p class="font-mono font-bold">' + data.audio_features.dynamics.dynamic_range.toFixed(1) + 'x</p></div><div><p class="text-slate-600">Komplexität</p><p class="font-mono font-bold">' + data.audio_features.rhythm_complexity.toFixed(1) + '/10</p></div></div></div>';

            // GESAMTBEWERTUNG
            const color = getScoreColor(data.overall_score);
            html += '<div class="bg-white rounded-2xl shadow-lg p-8 text-center mb-4"><h2 class="text-2xl font-bold mb-4">Gesamtbewertung</h2><div class="text-6xl font-bold text-' + color + '-600 mb-2">' + data.overall_score + '<span class="text-3xl text-gray-400">/10</span></div></div>';

            // FEEDBACK CATEGORIES
            const categories = [
                { title: 'Rhythmus & Timing', data: data.feedback.rhythm },
                { title: 'Harmonie', data: data.feedback.harmony },
                { title: 'Melodie & Phrasierung', data: data.feedback.melody },
                { title: 'Artikulation & Dynamik', data: data.feedback.articulation }
            ];
            categories.forEach(cat => {
                const catColor = getScoreColor(cat.data.score);
                html += '<div class="bg-white rounded-2xl shadow-lg p-6 mb-4"><div class="flex items-center justify-between mb-2"><h3 class="text-xl font-bold">' + cat.title + '</h3><span class="text-2xl font-bold text-' + catColor + '-600">' + cat.data.score.toFixed(1) + '</span></div><p class="text-gray-700 mb-3">' + cat.data.feedback + '</p><div class="bg-gray-50 rounded-lg p-4"><p class="font-semibold mb-2">Tipps:</p><ul class="space-y-1">' + cat.data.tips.map(tip => '<li class="text-sm text-gray-600">• ' + tip + '</li>').join('') + '</ul></div></div>';
            });

            document.getElementById('results').innerHTML = html;

            // Render Grand Staff
            setTimeout(() => {
                if (data.note_analysis && data.note_analysis.notes && data.note_analysis.notes.length > 0) {
                    const key = data.note_analysis.detected_scale || 'C Major';
                    const chords = data.note_analysis.chords || [];
                    const tempo = data.audio_features.tempo || 120;
                    const abcNotation = generateGrandStaffAbc(data.note_analysis.notes, chords, key, tempo);
                    console.log('ABC:', abcNotation.replace(/\\\\n/g, '\\n'));
                    
                    if (typeof ABCJS !== 'undefined') {
                        try {
                            ABCJS.renderAbc('pianoSheet', abcNotation.replace(/\\\\n/g, '\\n'), {
                                responsive: 'resize',
                                staffwidth: 700,
                                paddingtop: 20,
                                paddingbottom: 20
                            });
                        } catch (e) {
                            console.error('ABCJS Error:', e);
                            document.getElementById('pianoSheet').innerHTML = '<p class="text-red-500">Fehler beim Rendern: ' + e.message + '</p>';
                        }
                    }
                }
            }, 200);
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE

@app.get("/ai-status")
async def ai_status():
    return {"ai_enabled": apertus_client is not None, "model": "swiss-ai/Apertus-70B-Instruct-2509" if apertus_client else None, "note_detection": "MIDI Analysis"}

# ============================================================================
# JAZZ PATTERN ANALYSIS
# ============================================================================

def analyze_jazz_patterns(audio_features: Dict) -> Dict:
    tempo = audio_features.get("tempo", 120)
    rhythm_complexity = audio_features.get("rhythm_complexity", 5)
    note_density = audio_features.get("note_density", 2)
    
    if 60 <= tempo < 100: tempo_category, reference = "Ballad", "Ähnlich Bill Evans' 'Waltz for Debby'"
    elif 100 <= tempo < 140: tempo_category, reference = "Medium Swing", "Typisch für Miles Davis' 'So What'"
    elif 140 <= tempo < 200: tempo_category, reference = "Up-Tempo/Bebop", "Charlie Parker Bebop-Tempo"
    else: tempo_category, reference = "Very Fast", "John Coltrane 'Giant Steps' Tempo"
    
    if rhythm_complexity < 3: rhythm_assessment = "Einfache, klare Phrasierung"
    elif 3 <= rhythm_complexity < 6: rhythm_assessment = "Moderate Komplexität, gut balanciert"
    else: rhythm_assessment = "Sehr komplex, evtl. zu hektisch"
    
    if note_density < 2: density_assessment = "Sparsame Phrasierung (Monk-Stil)"
    elif 2 <= note_density < 4: density_assessment = "Ausgewogene Dichte (Mainstream Jazz)"
    else: density_assessment = "Sehr dichte Lines (Bebop/Coltrane-Stil)"
    
    artists = []
    if tempo < 100 and note_density < 2: artists = ["Bill Evans", "Keith Jarrett"]
    elif 140 <= tempo < 200 and note_density > 3: artists = ["Charlie Parker", "Dizzy Gillespie"]
    elif note_density > 4: artists = ["John Coltrane", "Michael Brecker"]
    else: artists = ["Miles Davis", "Dexter Gordon"]
    
    return {"tempo_category": tempo_category, "tempo_reference": reference, "rhythm_assessment": rhythm_assessment, "density_assessment": density_assessment, "swing_feel": "Classic Swing Feel" if 3 <= rhythm_complexity < 6 else "Complex Swing", "similar_artists": artists}

# ============================================================================
# APERTUS AI FEEDBACK
# ============================================================================

async def get_apertus_feedback(audio_features: Dict, jazz_analysis: Dict, note_analysis: Dict) -> Dict:
    if not apertus_client: return None
    
    try:
        try:
            kb = get_knowledge_base()
            jazz_context = kb.get_context_for_analysis(tempo=audio_features.get('tempo', 120), tempo_category=jazz_analysis['tempo_category'], rhythm_complexity=audio_features.get('rhythm_complexity', 5))
        except: jazz_context = ""
        
        chord_info = ""
        if note_analysis and note_analysis.get("chords"):
            chord_symbols = [c.get('symbol', '?') for c in note_analysis['chords'][:10]]
            chord_info = f"\n- Erkannte Akkorde: {' → '.join(chord_symbols)}"
            if note_analysis.get('progression'):
                prog = note_analysis['progression']
                if prog.get('type') and prog['type'] != 'Custom': chord_info += f"\n- Progression: {prog['type']}"
                if prog.get('roman_numerals'): chord_info += f"\n- Roman Numerals: {' → '.join(prog['roman_numerals'][:10])}"
        
        note_info = ""
        if note_analysis and note_analysis.get("total_notes", 0) > 0:
            note_info = f"""
MIDI ANALYSE (100% akkurat):
- Erkannte Noten: {note_analysis['total_notes']}
- Tonumfang: {note_analysis['pitch_range']['min_note']} bis {note_analysis['pitch_range']['max_note']}
- Häufigste Noten: {', '.join(note_analysis['most_common_notes'][:5])}
- Erkannte Tonart: {note_analysis.get('detected_scale', 'unbekannt')}{chord_info}"""
        
        prompt = f"""Du bist ein erfahrener Jazz-Lehrer. Analysiere diese Jazz-Improvisation:

MIDI-DATEN:
- Dauer: {audio_features.get('duration', 0):.1f}s, Tempo: {audio_features.get('tempo', 120):.1f} BPM
{note_info}

JAZZ-KONTEXT: {jazz_analysis['tempo_category']}, {jazz_analysis['rhythm_assessment']}
{jazz_context}

Gib Feedback in 4 Kategorien (je 1-10):
1. Rhythmus & Timing 2. Harmonie 3. Melodie & Phrasierung 4. Artikulation & Dynamik

Antworte NUR mit JSON (keine Backticks):
{{"rhythm": {{"score": 7.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "harmony": {{"score": 8.0, "feedback": "...", "tips": ["...", "...", "..."]}}, "melody": {{"score": 6.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "articulation": {{"score": 7.0, "feedback": "...", "tips": ["...", "...", "..."]}}}}"""
        
        response = apertus_client.chat_completion(model="swiss-ai/Apertus-70B-Instruct-2509", messages=[{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.7)
        text = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        return json.loads(text)
    except Exception as e:
        print(f"Apertus Error: {e}")
        return None

def generate_rule_based_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    tempo_stability = audio_features.get("tempo_stability", 0.5)
    rhythm_complexity = audio_features.get("rhythm_complexity", 5)
    dynamic_range = audio_features.get("dynamics", {}).get("dynamic_range", 0.5)
    note_density = audio_features.get("note_density", 2)
    
    return {
        "rhythm": {"score": round(min(10, max(1, (tempo_stability * 5) + (5 if 3 <= rhythm_complexity <= 6 else 3))), 1), "feedback": f"Timing-Stabilität bei {tempo_stability:.0%}. {jazz_analysis['rhythm_assessment']}.", "tips": [f"Übe mit Metronom bei {audio_features.get('tempo', 120):.0f} BPM", f"Höre {jazz_analysis['similar_artists'][0]}", f"Arbeite am {jazz_analysis['swing_feel']}"]},
        "harmony": {"score": 7.0, "feedback": "Harmonische Struktur erkannt.", "tips": ["Achte auf Guide Tones (3 und 7)", "Nutze chromatische Approach-Töne", "Studiere Bebop-Scales"]},
        "melody": {"score": 6.5 if 2 <= note_density <= 4 else 5.5, "feedback": f"{jazz_analysis['density_assessment']}.", "tips": ["Variiere Phrasenlängen", "Nutze mehr Pausen", f"Studiere {', '.join(jazz_analysis['similar_artists'][:2])}"]},
        "articulation": {"score": round(min(10, max(1, dynamic_range * 2 + 5)), 1), "feedback": f"Dynamik-Variation vorhanden. {jazz_analysis['density_assessment']}.", "tips": ["Arbeite an dynamischen Kontrasten", "Nutze Akzente", "Experimentiere mit Anschlagstärken"]}
    }

# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_midi_in_background(analysis_id: str, tmp_path: str):
    import asyncio, gc
    try:
        analysis_results[analysis_id] = {"status": "processing", "stage": "notes"}
        note_analysis = analyze_midi_file(tmp_path)
        if note_analysis.get('error') or note_analysis.get('total_notes', 0) == 0:
            raise Exception(f"MIDI failed: {note_analysis.get('error', 'No notes')}")
        
        duration = max(1, note_analysis.get('duration', 1))
        audio_features = {
            "duration": duration, "tempo": note_analysis.get('tempo_bpm', 120),
            "tempo_stability": note_analysis.get('timing', {}).get('precision_score', 0.8),
            "note_density": note_analysis.get('total_notes', 0) / duration,
            "pitch_range": {"min": note_analysis.get('pitch_range', {}).get('min', 0), "max": note_analysis.get('pitch_range', {}).get('max', 0)},
            "dynamics": {"mean_rms": note_analysis.get('dynamics', {}).get('mean', 80) / 127, "dynamic_range": note_analysis.get('dynamics', {}).get('range', 40) / 127},
            "rhythm_complexity": min(10, max(1, note_analysis.get('timing', {}).get('std_interval', 0.5) * 10)),
        }
        
        jazz_analysis = analyze_jazz_patterns(audio_features)
        gc.collect()
        
        analysis_results[analysis_id] = {"status": "processing", "stage": "ai"}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        feedback = loop.run_until_complete(get_apertus_feedback(audio_features, jazz_analysis, note_analysis))
        loop.close()
        
        if not feedback: feedback = generate_rule_based_feedback(audio_features, jazz_analysis)
        
        overall_score = (feedback["rhythm"]["score"] + feedback["harmony"]["score"] + feedback["melody"]["score"] + feedback["articulation"]["score"]) / 4
        
        analysis_results[analysis_id] = {"status": "completed", "result": {
            "overall_score": round(overall_score, 1), "audio_features": audio_features,
            "jazz_analysis": jazz_analysis, "note_analysis": note_analysis,
            "feedback": feedback, "ai_generated": apertus_client is not None
        }}
    except Exception as e:
        import traceback; traceback.print_exc()
        analysis_results[analysis_id] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/analyze-async")
async def analyze_midi_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not (file.filename.endswith('.mid') or file.filename.endswith('.midi')):
        raise HTTPException(status_code=400, detail="Nur MIDI-Dateien erlaubt")
    analysis_id = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    background_tasks.add_task(process_midi_in_background, analysis_id, tmp_path)
    return {"analysis_id": analysis_id, "status": "processing"}

@app.get("/result/{analysis_id}")
async def get_result(analysis_id: str):
    if analysis_id not in analysis_results: raise HTTPException(status_code=404, detail="Not found")
    return analysis_results[analysis_id]

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ai_enabled": apertus_client is not None, "analysis_type": "MIDI Grand Staff"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
