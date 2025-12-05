# Jazz Improvisation Feedback Platform - WITH APERTUS API
# FastAPI + Librosa + Apertus (Swiss AI from ETH/EPFL)

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
from scipy import signal
import tempfile
import os
from typing import Dict, List, Optional
import json
import uuid
from datetime import datetime
from huggingface_hub import InferenceClient

app = FastAPI(title="Jazz Feedback API")

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
            apertus_client = InferenceClient(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token
            )
            print("✅ Apertus API connected!")
        else:
            print("⚠️  HF_TOKEN not found - AI disabled")
    except Exception as e:
        print(f"⚠️  Apertus init failed: {e}")

initialize_apertus()

# In-Memory Storage
analysis_results = {}

# ============================================================================
# WEB UI
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback - Apertus AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
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
            <p class="text-gray-600">🇨🇭 Powered by Apertus - Swiss AI from ETH/EPFL</p>
        </div>

        <div id="aiStatus" class="mb-6"></div>

        <div class="bg-white rounded-2xl shadow-lg p-8 mb-6">
            <input type="file" id="fileInput" accept="audio/*" class="hidden">
            <div id="dropzone" class="border-3 border-dashed border-red-300 rounded-xl p-12 text-center cursor-pointer hover:border-red-500 hover:bg-red-50 transition-all">
                <svg class="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                <p class="text-lg font-semibold text-gray-700 mb-2" id="fileName">Audio-Datei hochladen</p>
                <p class="text-sm text-gray-500">MP3, WAV, M4A - bis zu 5 Minuten</p>
            </div>

            <div id="audioPlayerContainer" class="mt-6 hidden">
                <audio id="audioPlayer" controls class="w-full"></audio>
            </div>

            <button id="analyzeBtn" class="w-full mt-6 bg-red-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-red-700 disabled:bg-gray-400 transition-colors hidden">
                🎵 KI-Analyse starten
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

        fetch('/ai-status')
            .then(r => r.json())
            .then(data => {
                const statusDiv = document.getElementById('aiStatus');
                if (data.ai_enabled) {
                    statusDiv.innerHTML = '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="flex items-center gap-3"><svg class="w-8 h-8 text-red-600" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg><div><div class="text-sm font-medium text-red-900">🇨🇭 Apertus AI aktiv</div><div class="text-xs text-red-700">Swiss AI | ETH Zürich & EPFL</div></div></div></div>';
                } else {
                    statusDiv.innerHTML = '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">⚠️ AI deaktiviert - Nutze regel-basiertes Feedback</span></div>';
                }
            });

        document.getElementById('dropzone').addEventListener('click', () => {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('audio/')) {
                selectedFile = file;
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('audioPlayer').src = URL.createObjectURL(file);
                document.getElementById('audioPlayerContainer').classList.remove('hidden');
                document.getElementById('analyzeBtn').classList.remove('hidden');
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
                const response = await fetch('/analyze-async', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                analysisId = data.analysis_id;

                document.getElementById('loadingText').textContent = 'Analysiere Audio...';
                document.getElementById('progressBar').style.width = '30%';

                pollForResults(analysisId);

            } catch (error) {
                alert('Fehler: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
            }
        });

        async function pollForResults(id) {
            const maxAttempts = 120;
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
                        const progress = 30 + (attempts / maxAttempts) * 60;
                        document.getElementById('progressBar').style.width = progress + '%';
                        
                        if (data.stage === 'librosa') {
                            document.getElementById('loadingText').textContent = '🎵 Audio-Analyse...';
                        } else if (data.stage === 'ai') {
                            document.getElementById('loadingText').textContent = '🇨🇭 Apertus generiert Feedback...';
                        }
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
                } catch (error) {
                    console.error('Poll error:', error);
                }
            }, 2000);
        }

        function displayResults(data) {
            const getScoreColor = (score) => score >= 8 ? 'green' : score >= 6 ? 'yellow' : 'orange';
            let html = '';

            if (data.ai_generated) {
                html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-4 mb-6"><div class="flex items-center gap-2"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M13 7H7v6h6V7z"/></svg><span class="font-semibold">🇨🇭 Feedback von Apertus (Swiss AI)</span></div></div>';
            }

            html += '<div class="bg-gradient-to-br from-red-50 to-pink-50 border border-red-200 rounded-xl p-6"><h3 class="font-semibold text-red-900 mb-4">🎷 Jazz-Kontext</h3><div class="space-y-2 text-sm"><p><strong>Tempo:</strong> ' + data.jazz_analysis.tempo_category + '</p><p class="text-red-700">' + data.jazz_analysis.tempo_reference + '</p><p><strong>Rhythmik:</strong> ' + data.jazz_analysis.rhythm_assessment + '</p><p><strong>Dichte:</strong> ' + data.jazz_analysis.density_assessment + '</p><p><strong>Swing:</strong> ' + data.jazz_analysis.swing_feel + '</p><p><strong>Ähnlich:</strong> ' + data.jazz_analysis.similar_artists.join(', ') + '</p></div></div>';

            html += '<div class="bg-slate-50 border border-slate-200 rounded-xl p-6"><h3 class="font-semibold text-slate-900 mb-4">📊 Messwerte</h3><div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm"><div><p class="text-slate-600">Dauer</p><p class="font-mono font-bold">' + data.audio_features.duration.toFixed(1) + 's</p></div><div><p class="text-slate-600">Tempo</p><p class="font-mono font-bold">' + data.audio_features.tempo.toFixed(1) + ' BPM</p></div><div><p class="text-slate-600">Stabilität</p><p class="font-mono font-bold">' + (data.audio_features.tempo_stability * 100).toFixed(0) + '%</p></div><div><p class="text-slate-600">Noten-Dichte</p><p class="font-mono font-bold">' + data.audio_features.note_density.toFixed(2) + '/s</p></div><div><p class="text-slate-600">Dynamik</p><p class="font-mono font-bold">' + data.audio_features.dynamics.dynamic_range.toFixed(1) + 'x</p></div><div><p class="text-slate-600">Komplexität</p><p class="font-mono font-bold">' + data.audio_features.rhythm_complexity.toFixed(1) + '/10</p></div></div></div>';

            const color = getScoreColor(data.overall_score);
            html += '<div class="bg-white rounded-2xl shadow-lg p-8 text-center"><h2 class="text-2xl font-bold mb-4">Gesamtbewertung</h2><div class="text-6xl font-bold text-' + color + '-600 mb-2">' + data.overall_score + '<span class="text-3xl text-gray-400">/10</span></div></div>';

            const categories = [
                { title: 'Rhythmus & Timing', data: data.feedback.rhythm },
                { title: 'Harmonie', data: data.feedback.harmony },
                { title: 'Melodie & Phrasierung', data: data.feedback.melody },
                { title: 'Artikulation & Dynamik', data: data.feedback.articulation }
            ];

            categories.forEach(cat => {
                const catColor = getScoreColor(cat.data.score);
                html += '<div class="bg-white rounded-2xl shadow-lg p-6"><div class="flex items-center justify-between mb-2"><h3 class="text-xl font-bold">' + cat.title + '</h3><span class="text-2xl font-bold text-' + catColor + '-600">' + cat.data.score.toFixed(1) + '</span></div><p class="text-gray-700 mb-3">' + cat.data.feedback + '</p><div class="bg-gray-50 rounded-lg p-4"><p class="font-semibold mb-2">Tipps:</p><ul class="space-y-1">' + cat.data.tips.map(tip => '<li class="text-sm text-gray-600">• ' + tip + '</li>').join('') + '</ul></div></div>';
            });

            document.getElementById('results').innerHTML = html;
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
    return {
        "ai_enabled": apertus_client is not None,
        "model": "swiss-ai/Apertus-70B-Instruct-2509" if apertus_client else None
    }


# ============================================================================
# AUDIO ANALYSIS
# ============================================================================

def analyze_audio_file(audio_path: str) -> Dict:
    y, sr = librosa.load(audio_path, sr=11025, duration=600)
    duration = librosa.get_duration(y=y, sr=sr)
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo_stability = calculate_tempo_stability(beat_times)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_sequence = extract_pitch_sequence(pitches, magnitudes)
    
    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    rhythm_complexity = calculate_rhythm_complexity(onsets)
    
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = float(np.max(rms) / (np.mean(rms) + 1e-6))
    dynamic_variance = float(np.std(rms))
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    return {
        "duration": float(duration),
        "tempo": float(tempo),
        "tempo_stability": tempo_stability,
        "beats": len(beat_times),
        "onsets": len(onsets),
        "note_density": len(onsets) / duration,
        "pitch_range": {
            "min": float(np.min(pitch_sequence[pitch_sequence > 0])) if len(pitch_sequence[pitch_sequence > 0]) > 0 else 0,
            "max": float(np.max(pitch_sequence)),
            "mean": float(np.mean(pitch_sequence[pitch_sequence > 0])) if len(pitch_sequence[pitch_sequence > 0]) > 0 else 0
        },
        "dynamics": {
            "mean_rms": float(np.mean(rms)),
            "max_rms": float(np.max(rms)),
            "dynamic_range": dynamic_range,
            "variance": dynamic_variance
        },
        "spectral": {
            "centroid_mean": float(np.mean(spectral_centroids)),
            "centroid_std": float(np.std(spectral_centroids)),
            "rolloff_mean": float(np.mean(spectral_rolloff)),
            "flatness_mean": float(np.mean(spectral_flatness))
        },
        "rhythm_complexity": rhythm_complexity,
        "harmonic_content": float(np.mean(chroma))
    }


def calculate_tempo_stability(beat_times: np.ndarray) -> float:
    if len(beat_times) < 3:
        return 0.5
    intervals = np.diff(beat_times)
    stability = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-6))
    return float(np.clip(stability, 0, 1))


def extract_pitch_sequence(pitches: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
    pitch_sequence = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_sequence.append(pitch)
    return np.array(pitch_sequence)


def calculate_rhythm_complexity(onsets: np.ndarray) -> float:
    if len(onsets) < 3:
        return 1.0
    intervals = np.diff(onsets)
    complexity = np.std(intervals) / (np.mean(intervals) + 1e-6)
    return float(np.clip(complexity * 5, 0, 10))


def analyze_jazz_patterns(audio_features: Dict) -> Dict:
    tempo = audio_features["tempo"]
    rhythm_complexity = audio_features["rhythm_complexity"]
    note_density = audio_features["note_density"]
    
    if 60 <= tempo < 100:
        tempo_category = "Ballad"
        reference = "Ähnlich Bill Evans' 'Waltz for Debby'"
    elif 100 <= tempo < 140:
        tempo_category = "Medium Swing"
        reference = "Typisch für Miles Davis' 'So What'"
    elif 140 <= tempo < 200:
        tempo_category = "Up-Tempo/Bebop"
        reference = "Charlie Parker Bebop-Tempo"
    else:
        tempo_category = "Very Fast"
        reference = "John Coltrane 'Giant Steps' Tempo"
    
    if rhythm_complexity < 3:
        rhythm_assessment = "Einfache, klare Phrasierung"
    elif 3 <= rhythm_complexity < 6:
        rhythm_assessment = "Moderate Komplexität, gut balanciert"
    else:
        rhythm_assessment = "Sehr komplex, evtl. zu hektisch"
    
    if note_density < 2:
        density_assessment = "Sparsame Phrasierung (Monk-Stil)"
    elif 2 <= note_density < 4:
        density_assessment = "Ausgewogene Dichte (Mainstream Jazz)"
    else:
        density_assessment = "Sehr dichte Lines (Bebop/Coltrane-Stil)"
    
    artists = []
    if tempo < 100 and note_density < 2:
        artists.extend(["Bill Evans", "Keith Jarrett"])
    elif 140 <= tempo < 200 and note_density > 3:
        artists.extend(["Charlie Parker", "Dizzy Gillespie"])
    elif note_density > 4:
        artists.extend(["John Coltrane", "Michael Brecker"])
    else:
        artists.extend(["Miles Davis", "Dexter Gordon"])
    
    return {
        "tempo_category": tempo_category,
        "tempo_reference": reference,
        "rhythm_assessment": rhythm_assessment,
        "density_assessment": density_assessment,
        "swing_feel": "Classic Swing Feel" if 3 <= rhythm_complexity < 6 else "Complex Swing",
        "similar_artists": artists
    }


# ============================================================================
# APERTUS AI FEEDBACK
# ============================================================================

async def get_apertus_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    """Nutzt Apertus AI für intelligentes Feedback"""
    
    if not apertus_client:
        print("Apertus nicht verfügbar, fallback")
        return None
    
    try:
        prompt = f"""Du bist ein erfahrener Jazz-Lehrer mit 30 Jahren Unterrichtserfahrung. Analysiere diese Jazz-Improvisation und gib konstruktives, spezifisches Feedback.

AUDIO-DATEN (Librosa-Analyse):
- Dauer: {audio_features['duration']:.1f} Sekunden
- Tempo: {audio_features['tempo']:.1f} BPM
- Tempo-Stabilität: {audio_features['tempo_stability']:.1%}
- Noten-Dichte: {audio_features['note_density']:.2f} Noten/Sekunde
- Rhythmische Komplexität: {audio_features['rhythm_complexity']:.1f}/10
- Dynamik-Range: {audio_features['dynamics']['dynamic_range']:.1f}x
- Spektraler Centroid: {audio_features['spectral']['centroid_mean']:.0f} Hz

JAZZ-KONTEXT:
- Tempo-Kategorie: {jazz_analysis['tempo_category']}
- Referenz: {jazz_analysis['tempo_reference']}
- Rhythmische Bewertung: {jazz_analysis['rhythm_assessment']}
- Phrasierungs-Dichte: {jazz_analysis['density_assessment']}
- Swing-Feel: {jazz_analysis['swing_feel']}
- Ähnliche Künstler: {', '.join(jazz_analysis['similar_artists'])}

Gib detailliertes Feedback in 4 Kategorien (je 1-10 Punkte):
1. Rhythmus & Timing - Analysiere Tempo-Stabilität, Swing-Feel, rhythmische Komplexität
2. Harmonie - Bewerte basierend auf spektralen Eigenschaften und Jazz-Kontext
3. Melodie & Phrasierung - Beurteile Noten-Dichte, Phrasenlänge, melodische Entwicklung
4. Artikulation & Dynamik - Analysiere Dynamik-Range und Variabilität

Für jede Kategorie:
- Gib einen Score (1.0 bis 10.0)
- Schreibe 2-3 Sätze spezifisches Feedback
- Gib 3 konkrete, umsetzbare Übungstipps

Antworte NUR mit diesem JSON-Format (keine Markdown-Backticks):
{{
  "rhythm": {{
    "score": 7.5,
    "feedback": "Dein Tempo ist...",
    "tips": ["Tipp 1", "Tipp 2", "Tipp 3"]
  }},
  "harmony": {{
    "score": 8.0,
    "feedback": "Die harmonische...",
    "tips": ["Tipp 1", "Tipp 2", "Tipp 3"]
  }},
  "melody": {{
    "score": 6.5,
    "feedback": "Deine melodischen...",
    "tips": ["Tipp 1", "Tipp 2", "Tipp 3"]
  }},
  "articulation": {{
    "score": 7.0,
    "feedback": "Die Artikulation...",
    "tips": ["Tipp 1", "Tipp 2", "Tipp 3"]
  }}
}}"""
        
        print("🇨🇭 Calling Apertus API...")
        
        response = apertus_client.chat_completion(
            model="swiss-ai/Apertus-70B-Instruct-2509",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        
        text = response.choices[0].message.content
        clean = text.replace('```json', '').replace('```', '').strip()
        
        # Find JSON
        if "{" in clean:
            json_start = clean.find("{")
            json_end = clean.rfind("}") + 1
            clean = clean[json_start:json_end]
        
        feedback = json.loads(clean)
        print("✅ Apertus feedback generated")
        return feedback
        
    except Exception as e:
        print(f"Apertus API Fehler: {e}")
        return None


def generate_rule_based_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    """Fallback: Regel-basiertes Feedback"""
    tempo_stability = audio_features["tempo_stability"]
    rhythm_complexity = audio_features["rhythm_complexity"]
    dynamic_range = audio_features["dynamics"]["dynamic_range"]
    note_density = audio_features["note_density"]
    
    rhythm_score = (tempo_stability * 5) + (5 if 3 <= rhythm_complexity <= 6 else 3)
    rhythm_feedback = f"Tempo-Stabilität bei {tempo_stability:.0%}. {jazz_analysis['rhythm_assessment']}. {jazz_analysis['tempo_reference']}."
    
    articulation_score = min(10, dynamic_range * 2)
    articulation_feedback = f"Dynamik-Range von {dynamic_range:.1f}x. {jazz_analysis['density_assessment']}."
    
    return {
        "rhythm": {
            "score": round(rhythm_score, 1),
            "feedback": rhythm_feedback,
            "tips": [
                f"Übe mit Metronom bei {audio_features['tempo']:.0f} BPM",
                "Höre dir " + jazz_analysis['similar_artists'][0] + " für rhythmische Inspiration an",
                f"Arbeite am {jazz_analysis['swing_feel']}"
            ]
        },
        "harmony": {
            "score": 7.0,
            "feedback": f"Harmonische Inhalte erkannt. Spektraler Centroid bei {audio_features['spectral']['centroid_mean']:.0f} Hz.",
            "tips": [
                "Experimentiere mit ii-V-I Progressionen",
                "Nutze chromatische Approach-Töne",
                "Studiere Bebop-Scales"
            ]
        },
        "melody": {
            "score": 6.5 if 2 <= note_density <= 4 else 5.5,
            "feedback": f"{jazz_analysis['density_assessment']}. Noten-Dichte: {note_density:.2f}/Sekunde.",
            "tips": [
                "Variiere Phrasenlängen",
                "Nutze mehr Pausen zwischen Phrasen",
                f"Studiere {', '.join(jazz_analysis['similar_artists'][:2])}"
            ]
        },
        "articulation": {
            "score": round(articulation_score, 1),
            "feedback": articulation_feedback,
            "tips": [
                "Arbeite an dynamischen Kontrasten",
                "Nutze Akzente für rhythmische Betonung",
                "Experimentiere mit Ghost Notes"
            ]
        }
    }


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_audio_in_background(analysis_id: str, tmp_path: str):
    """Background task"""
    import asyncio
    
    try:
        analysis_results[analysis_id] = {"status": "processing", "stage": "librosa"}
        
        # Step 1: Librosa
        audio_features = analyze_audio_file(tmp_path)
        jazz_analysis = analyze_jazz_patterns(audio_features)
        
        # Step 2: Apertus AI (versuche AI, fallback zu regelbasiert)
        analysis_results[analysis_id] = {"status": "processing", "stage": "ai"}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        feedback = loop.run_until_complete(
            get_apertus_feedback(audio_features, jazz_analysis)
        )
        loop.close()
        
        if not feedback:
            feedback = generate_rule_based_feedback(audio_features, jazz_analysis)
        
        overall_score = (
            feedback["rhythm"]["score"] +
            feedback["harmony"]["score"] +
            feedback["melody"]["score"] +
            feedback["articulation"]["score"]
        ) / 4
        
        result = {
            "overall_score": round(overall_score, 1),
            "audio_features": audio_features,
            "jazz_analysis": jazz_analysis,
            "feedback": feedback,
            "ai_generated": apertus_client is not None
        }
        
        analysis_results[analysis_id] = {
            "status": "completed",
            "result": result
        }
        
    except Exception as e:
        analysis_results[analysis_id] = {
            "status": "error",
            "error": str(e)
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/analyze-async")
async def analyze_audio_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Nur Audio-Dateien erlaubt")
    
    analysis_id = str(uuid.uuid4())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    background_tasks.add_task(process_audio_in_background, analysis_id, tmp_path)
    
    return {"analysis_id": analysis_id, "status": "processing"}


@app.get("/result/{analysis_id}")
async def get_result(analysis_id: str):
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_enabled": apertus_client is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
