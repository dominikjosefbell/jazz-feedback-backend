# Jazz Improvisation Feedback Platform - MIDI VERSION
# With Key Selector + Grand Staff Notation

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

apertus_client = None

def initialize_apertus():
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

analysis_results = {}

# ============================================================================
# WEB UI WITH KEY SELECTOR
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback - MIDI Analyse</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
            
            <!-- File Upload -->
            <div id="dropzone" class="border-3 border-dashed border-red-300 rounded-xl p-12 text-center cursor-pointer hover:border-red-500 hover:bg-red-50 transition-all mb-6">
                <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg font-semibold text-gray-700 mb-2" id="fileName">MIDI-Datei hochladen</p>
                <p class="text-sm text-gray-500">MIDI-Dateien (.mid, .midi)</p>
            </div>

            <!-- Key Selector - CLEAN VERSION -->
            <div id="keySelector" class="hidden mb-6">
                <label class="block text-sm font-semibold text-gray-700 mb-2">🎼 Tonart:</label>
                <select id="keySelect" class="w-full p-3 border-2 border-gray-200 rounded-xl focus:border-red-500 focus:ring-2 focus:ring-red-200 transition-all text-lg">
                    <optgroup label="Dur">
                        <option value="C Major">C-Dur</option>
                        <option value="G Major">G-Dur</option>
                        <option value="D Major">D-Dur</option>
                        <option value="A Major">A-Dur</option>
                        <option value="E Major">E-Dur</option>
                        <option value="B Major">H-Dur</option>
                        <option value="F# Major">Fis-Dur</option>
                        <option value="F Major">F-Dur</option>
                        <option value="Bb Major">B-Dur</option>
                        <option value="Eb Major">Es-Dur</option>
                        <option value="Ab Major">As-Dur</option>
                        <option value="Db Major">Des-Dur</option>
                    </optgroup>
                    <optgroup label="Moll">
                        <option value="A Minor">a-Moll</option>
                        <option value="E Minor">e-Moll</option>
                        <option value="B Minor">h-Moll</option>
                        <option value="F# Minor">fis-Moll</option>
                        <option value="D Minor">d-Moll</option>
                        <option value="G Minor">g-Moll</option>
                        <option value="C Minor">c-Moll</option>
                        <option value="F Minor">f-Moll</option>
                    </optgroup>
                </select>
            </div>

            <button id="analyzeBtn" class="w-full bg-red-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-red-700 disabled:bg-gray-400 transition-colors hidden">
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
                statusDiv.innerHTML = '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="flex items-center gap-3"><div class="text-sm font-medium text-red-900">🇨🇭 Apertus AI + 🎹 MIDI + 🎼 Notenschrift aktiv</div></div></div>';
            } else {
                statusDiv.innerHTML = '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">⚠️ AI deaktiviert</span></div>';
            }
        });

        document.getElementById('dropzone').addEventListener('click', () => document.getElementById('fileInput').click());

        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && (file.name.endsWith('.mid') || file.name.endsWith('.midi'))) {
                selectedFile = file;
                document.getElementById('fileName').textContent = '✓ ' + file.name;
                document.getElementById('dropzone').classList.add('border-green-400', 'bg-green-50');
                document.getElementById('dropzone').classList.remove('border-red-300');
                document.getElementById('keySelector').classList.remove('hidden');
                document.getElementById('analyzeBtn').classList.remove('hidden');
            } else {
                alert('Bitte eine MIDI-Datei (.mid oder .midi) auswählen');
            }
        });

        document.getElementById('analyzeBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const selectedKey = document.getElementById('keySelect').value;
            
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').innerHTML = '';
            document.getElementById('loadingText').textContent = 'Uploading...';
            document.getElementById('progressBar').style.width = '10%';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('key', selectedKey);

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

        // ABC Key mapping
        function getAbcKey(userKey) {
            const map = {
                'C Major': 'C', 'G Major': 'G', 'D Major': 'D', 'A Major': 'A', 
                'E Major': 'E', 'B Major': 'B', 'F# Major': 'F#', 'F Major': 'F',
                'Bb Major': 'Bb', 'Eb Major': 'Eb', 'Ab Major': 'Ab', 'Db Major': 'Db',
                'A Minor': 'Am', 'E Minor': 'Em', 'B Minor': 'Bm', 'F# Minor': 'F#m',
                'D Minor': 'Dm', 'G Minor': 'Gm', 'C Minor': 'Cm', 'F Minor': 'Fm'
            };
            return map[userKey] || 'C';
        }

        // Simple MIDI to ABC - treble clef
        function midiToAbcTreble(pitch) {
            const notes = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B'];
            const oct = Math.floor(pitch / 12) - 1;
            const n = notes[pitch % 12];
            if (oct === 5) return n.toLowerCase();
            if (oct === 6) return n.toLowerCase() + "'";
            if (oct === 4) return n;
            if (oct === 3) return n + ",";
            return n;
        }

        // Simple MIDI to ABC - bass clef
        function midiToAbcBass(pitch) {
            const notes = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B'];
            const oct = Math.floor(pitch / 12) - 1;
            const n = notes[pitch % 12];
            if (oct === 3) return n;
            if (oct === 2) return n + ",";
            if (oct === 1) return n + ",,";
            if (oct === 4) return n.toLowerCase();
            return n + ",";
        }

        // Generate ABC from CHORDS (not individual notes!)
        function generateAbcFromChords(chords, userKey, tempo) {
            if (!chords || chords.length === 0) return null;
            
            const abcKey = getAbcKey(userKey);
            const bpm = Math.round(tempo) || 120;
            
            let abc = "X:1\\n";
            abc += "T:Erkannte Akkorde\\n";
            abc += "M:4/4\\n";
            abc += "L:1/2\\n";
            abc += "Q:1/4=" + bpm + "\\n";
            abc += "K:" + abcKey + "\\n";
            abc += "%%staves {1 2}\\n";
            abc += "V:1 clef=treble\\n";
            abc += "V:2 clef=bass\\n";
            
            // Treble: notes >= 60 (middle C)
            abc += "[V:1] ";
            chords.forEach((chord, idx) => {
                // Add chord symbol
                abc += '"' + chord.symbol + '"';
                
                // Get treble notes (>= middle C)
                const trebleNotes = chord.pitches.filter(p => p >= 60).sort((a,b) => a-b);
                
                if (trebleNotes.length === 0) {
                    abc += "z2";
                } else if (trebleNotes.length === 1) {
                    abc += midiToAbcTreble(trebleNotes[0]) + "2";
                } else {
                    abc += "[";
                    trebleNotes.forEach(p => abc += midiToAbcTreble(p));
                    abc += "]2";
                }
                
                abc += " |";
            });
            abc += "|\\n";
            
            // Bass: notes < 60
            abc += "[V:2] ";
            chords.forEach((chord, idx) => {
                const bassNotes = chord.pitches.filter(p => p < 60).sort((a,b) => a-b);
                
                if (bassNotes.length === 0) {
                    abc += "z2";
                } else if (bassNotes.length === 1) {
                    abc += midiToAbcBass(bassNotes[0]) + "2";
                } else {
                    abc += "[";
                    bassNotes.forEach(p => abc += midiToAbcBass(p));
                    abc += "]2";
                }
                
                abc += " |";
            });
            abc += "|";
            
            return abc;
        }

        function displayResults(data) {
            const getScoreColor = (score) => score >= 8 ? 'green' : score >= 6 ? 'yellow' : 'orange';
            let html = '';
            
            const userKey = data.user_key || 'C Major';

            if (data.ai_generated) {
                html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-4 mb-6"><span class="font-semibold">🇨🇭 Feedback von Apertus AI</span></div>';
            }

            // SHEET MUSIC - Based on CHORDS
            if (data.note_analysis && data.note_analysis.chords && data.note_analysis.chords.length > 0) {
                html += '<div class="bg-white border-2 border-gray-200 rounded-2xl p-6 mb-4 shadow-lg">';
                html += '<h3 class="font-bold text-xl text-gray-900 mb-2">🎼 Notenschrift</h3>';
                html += '<p class="text-sm text-gray-500 mb-4">Tonart: <strong>' + userKey + '</strong></p>';
                html += '<div id="pianoSheet" class="bg-gray-50 rounded-xl p-4 overflow-x-auto min-h-[180px]"></div>';
                html += '</div>';
            }

            // CHORD DETECTION - Visual boxes
            if (data.note_analysis && data.note_analysis.chords && data.note_analysis.chords.length > 0) {
                html += '<div class="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-6 mb-4">';
                html += '<h3 class="font-semibold text-blue-900 mb-4">🎹 Erkannte Akkorde</h3>';
                html += '<div class="flex flex-wrap gap-3 mb-4 items-center">';
                data.note_analysis.chords.forEach((chord, index) => {
                    const isMinor = chord.type && chord.type.includes('m') && !chord.type.includes('maj');
                    const bgColor = isMinor ? 'bg-indigo-100 border-indigo-300' : 'bg-blue-100 border-blue-300';
                    html += '<div class="' + bgColor + ' border-2 rounded-xl px-4 py-3 text-center shadow-sm">';
                    html += '<div class="font-bold text-xl text-gray-800">' + chord.symbol + '</div>';
                    html += '<div class="text-sm text-gray-600 mt-1">' + chord.notes.join(' · ') + '</div>';
                    html += '</div>';
                    if (index < data.note_analysis.chords.length - 1) html += '<div class="text-gray-400 text-2xl">→</div>';
                });
                html += '</div>';
                
                // Progression info
                if (data.note_analysis.progression) {
                    html += '<div class="bg-white/60 rounded-lg p-4 mt-3">';
                    if (data.note_analysis.progression.type && data.note_analysis.progression.type !== 'Custom') {
                        html += '<p class="text-green-700 font-semibold">✓ ' + data.note_analysis.progression.type + '</p>';
                    }
                    if (data.note_analysis.progression.roman_numerals && data.note_analysis.progression.roman_numerals.length > 0) {
                        html += '<p class="text-sm text-blue-600 mt-1">' + data.note_analysis.progression.roman_numerals.join(' → ') + '</p>';
                    }
                    html += '</div>';
                }
                html += '</div>';
            }

            // NOTE DETECTION
            if (data.note_analysis && data.note_analysis.total_notes > 0) {
                html += '<div class="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6 mb-4">';
                html += '<h3 class="font-semibold text-purple-900 mb-4">📊 Analyse</h3>';
                html += '<div class="grid grid-cols-2 gap-4 text-sm">';
                html += '<div><strong>Noten:</strong> ' + data.note_analysis.total_notes + '</div>';
                html += '<div><strong>Tonumfang:</strong> ' + data.note_analysis.pitch_range.min_note + ' - ' + data.note_analysis.pitch_range.max_note + '</div>';
                html += '<div><strong>Tempo:</strong> ' + data.audio_features.tempo.toFixed(0) + ' BPM</div>';
                html += '<div><strong>Dauer:</strong> ' + data.audio_features.duration.toFixed(1) + 's</div>';
                html += '</div></div>';
            }

            // JAZZ CONTEXT
            html += '<div class="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6 mb-4"><h3 class="font-semibold text-red-900 mb-4">🎷 Jazz-Kontext</h3><div class="space-y-2 text-sm"><p><strong>Tempo:</strong> ' + data.jazz_analysis.tempo_category + '</p><p class="text-red-700">' + data.jazz_analysis.tempo_reference + '</p><p><strong>Ähnlich:</strong> ' + data.jazz_analysis.similar_artists.join(', ') + '</p></div></div>';

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

            // Render sheet music from CHORDS
            setTimeout(() => {
                if (data.note_analysis && data.note_analysis.chords && data.note_analysis.chords.length > 0) {
                    const tempo = data.audio_features.tempo || 120;
                    const abcNotation = generateAbcFromChords(data.note_analysis.chords, userKey, tempo);
                    
                    if (abcNotation && typeof ABCJS !== 'undefined') {
                        console.log('ABC:', abcNotation.replace(/\\\\n/g, '\\n'));
                        try {
                            ABCJS.renderAbc('pianoSheet', abcNotation.replace(/\\\\n/g, '\\n'), {
                                responsive: 'resize',
                                staffwidth: 600,
                                paddingtop: 10,
                                paddingbottom: 10
                            });
                        } catch (e) {
                            console.error('ABCJS Error:', e);
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
    return {"ai_enabled": apertus_client is not None}

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
    if not apertus_client: return None
    
    try:
        try:
            kb = get_knowledge_base()
            jazz_context = kb.get_context_for_analysis(tempo=audio_features.get('tempo', 120), tempo_category=jazz_analysis['tempo_category'], rhythm_complexity=5)
        except: jazz_context = ""
        
        chord_info = ""
        if note_analysis and note_analysis.get("chords"):
            chord_symbols = [c.get('symbol', '?') for c in note_analysis['chords']]
            chord_info = f"Akkorde: {' → '.join(chord_symbols)}"
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
        
        response = apertus_client.chat_completion(model="swiss-ai/Apertus-70B-Instruct-2509", messages=[{"role": "user", "content": prompt}], max_tokens=1200, temperature=0.7)
        text = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
        if "{" in text: text = text[text.find("{"):text.rfind("}")+1]
        return json.loads(text)
    except Exception as e:
        print(f"Apertus Error: {e}")
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
            "ai_generated": apertus_client is not None,
            "user_key": user_key
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
    return {"status": "healthy", "ai_enabled": apertus_client is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
