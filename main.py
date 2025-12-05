# Jazz Improvisation Feedback Platform - WITH BASIC PITCH (MEMORY OPTIMIZED)
# FastAPI + Librosa + Basic Pitch + Apertus AI

from knowledge_loader import get_knowledge_base
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
# NOTE: basic_pitch imported only when needed to save memory

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
# WEB UI (unchanged, just note the new features in status)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jazz Feedback - Apertus AI + Basic Pitch</title>
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
            <p class="text-gray-600">🇨🇭 Apertus AI + 🎹 Basic Pitch Note Detection</p>
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
                🎵 Analyse starten (mit Note-Detection)
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
                    statusDiv.innerHTML = '<div class="bg-gradient-to-r from-red-50 to-white border border-red-200 rounded-xl p-4"><div class="flex items-center gap-3"><svg class="w-8 h-8 text-red-600" viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg><div><div class="text-sm font-medium text-red-900">🇨🇭 Apertus AI + 🎹 Basic Pitch aktiv</div><div class="text-xs text-red-700">Note Detection & Swiss AI Feedback</div></div></div></div>';
                } else {
                    statusDiv.innerHTML = '<div class="bg-yellow-50 border border-yellow-200 rounded-xl p-4"><span class="text-sm font-medium text-yellow-900">⚠️ AI deaktiviert - Nutze regel-basiertes Feedback (Note Detection aktiv)</span></div>';
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
                document.getElementById('progressBar').style.width = '20%';

                pollForResults(analysisId);

            } catch (error) {
                alert('Fehler: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
            }
        });

        async function pollForResults(id) {
            const maxAttempts = 60; // 60 attempts × 5 seconds = 5 minutes
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
                        
                        if (data.stage === 'librosa') {
                            document.getElementById('loadingText').textContent = '🎵 Librosa Analyse...';
                        } else if (data.stage === 'notes') {
                            document.getElementById('loadingText').textContent = '🎹 Note Detection (kann bis zu 2 Min dauern)...';
                        } else if (data.stage === 'ai') {
                            document.getElementById('loadingText').textContent = '🇨🇭 Apertus AI Feedback...';
                        }
                    } else if (data.status === 'error') {
                        clearInterval(interval);
                        alert('Fehler: ' + data.error);
                        document.getElementById('loading').classList.add('hidden');
                    }

                    if (attempts >= maxAttempts) {
                        clearInterval(interval);
                        alert('Timeout: Analyse dauert zu lange. Bitte versuche eine kürzere Audio-Datei oder warte noch etwas und lade die Seite neu.');
                        document.getElementById('loading').classList.add('hidden');
                    }
                } catch (error) {
                    console.error('Poll error:', error);
                }
            }, 5000); // Check every 5 seconds instead of 2
        }

        function displayResults(data) {
            const getScoreColor = (score) => score >= 8 ? 'green' : score >= 6 ? 'yellow' : 'orange';
            let html = '';

            if (data.ai_generated) {
                html += '<div class="bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl p-4 mb-6"><div class="flex items-center gap-2"><svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M13 7H7v6h6V7z"/></svg><span class="font-semibold">🇨🇭 Feedback von Apertus + 🎹 Note Analysis</span></div></div>';
            }

            // Note Detection Results
            if (data.note_analysis) {
                html += '<div class="bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-200 rounded-xl p-6"><h3 class="font-semibold text-purple-900 mb-4">🎹 Note Detection</h3><div class="space-y-2 text-sm">';
                html += '<p><strong>Erkannte Noten:</strong> ' + data.note_analysis.total_notes + '</p>';
                html += '<p><strong>Tonumfang:</strong> ' + data.note_analysis.pitch_range.min_note + ' - ' + data.note_analysis.pitch_range.max_note + '</p>';
                html += '<p><strong>Häufigste Noten:</strong> ' + data.note_analysis.most_common_notes.join(', ') + '</p>';
                if (data.note_analysis.detected_scale) {
                    html += '<p><strong>Vermutete Tonart:</strong> ' + data.note_analysis.detected_scale + '</p>';
                }
                html += '</div></div>';
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
        "model": "swiss-ai/Apertus-70B-Instruct-2509" if apertus_client else None,
        "note_detection": "Basic Pitch (Spotify)"
    }


# ============================================================================
# AUDIO ANALYSIS (Librosa - unchanged)
# ============================================================================

def analyze_audio_file(audio_path: str) -> Dict:
    """Librosa audio analysis (Memory Optimized)"""
    # Load with lower sample rate to save memory
    y, sr = librosa.load(audio_path, sr=11025, duration=600, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo_stability = calculate_tempo_stability(beat_times)
    
    # Free beat data immediately
    del beats
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=1024)  # Smaller FFT
    pitch_sequence = extract_pitch_sequence(pitches, magnitudes)
    
    # Free pitch data
    del pitches
    del magnitudes
    
    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, n_chroma=12, n_octaves=5)  # Reduce octaves
    
    # Free HPSS data
    del harmonic
    del percussive
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    rhythm_complexity = calculate_rhythm_complexity(onsets)
    
    # Free onset data
    del onset_env
    
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]  # Smaller frames
    dynamic_range = float(np.max(rms) / (np.mean(rms) + 1e-6))
    dynamic_variance = float(np.std(rms))
    
    # Free RMS
    del rms
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024)[0]
    
    result = {
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
            "mean_rms": float(np.mean([dynamic_range])),
            "max_rms": float(dynamic_range),
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
    
    # Free all remaining arrays
    del y
    del beat_times
    del pitch_sequence
    del chroma
    del onsets
    del spectral_centroids
    del spectral_rolloff
    del spectral_flatness
    
    return result


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
# NOTE DETECTION (Basic Pitch - Memory Optimized)
# ============================================================================

def analyze_notes_with_basic_pitch(audio_path: str) -> Dict:
    """
    Use Basic Pitch to extract notes from audio
    MEMORY OPTIMIZED: Loads model only when needed, cleans up after
    """
    import gc
    
    try:
        print("🎹 Running Basic Pitch note detection (memory optimized)...")
        
        # Import only when needed (lazy loading)
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        
        # Load audio with lower sample rate to save memory
        import librosa as lb
        audio_data, sr = lb.load(audio_path, sr=22050, mono=True)  # Lower SR = less memory
        
        # Run Basic Pitch prediction with memory constraints
        model_output, midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH
        )
        
        # Process results immediately and free memory
        notes = []
        for start_time, end_time, pitch, amplitude in note_events[:200]:  # Limit to 200 notes
            note_name = lb.midi_to_note(int(pitch))
            notes.append({
                "start": float(start_time),
                "end": float(end_time),
                "pitch": int(pitch),
                "note": note_name,
                "amplitude": float(amplitude)
            })
        
        # Extract statistics
        pitches = [n["pitch"] for n in notes]
        note_names = [n["note"] for n in notes]
        
        # Most common notes
        from collections import Counter
        note_counter = Counter(note_names)
        most_common = note_counter.most_common(5)
        
        # Pitch range
        min_pitch = min(pitches) if pitches else 0
        max_pitch = max(pitches) if pitches else 0
        
        # Simple scale detection
        detected_scale = detect_scale_simple(note_names)
        
        result = {
            "total_notes": len(note_events),
            "notes": notes[:50],  # Only return first 50 to save memory
            "pitch_range": {
                "min": min_pitch,
                "max": max_pitch,
                "min_note": lb.midi_to_note(min_pitch) if min_pitch > 0 else "N/A",
                "max_note": lb.midi_to_note(max_pitch) if max_pitch > 0 else "N/A"
            },
            "most_common_notes": [note for note, count in most_common],
            "detected_scale": detected_scale
        }
        
        # CRITICAL: Free memory immediately
        del model_output
        del midi_data
        del note_events
        del audio_data
        del notes
        del pitches
        del note_names
        gc.collect()
        
        print("✅ Basic Pitch completed, memory freed")
        return result
        
    except Exception as e:
        print(f"Basic Pitch error: {e}")
        # Clean up on error too
        gc.collect()
        return {
            "total_notes": 0,
            "error": str(e)
        }


def detect_scale_simple(note_names: List[str]) -> Optional[str]:
    """
    Simple scale detection based on note frequency
    Returns likely key/scale
    """
    if not note_names:
        return None
    
    # Count note occurrences (ignore octaves)
    from collections import Counter
    notes_no_octave = [n[:-1] if len(n) > 1 else n for n in note_names]
    counter = Counter(notes_no_octave)
    
    # Most common note is likely tonic
    most_common = counter.most_common(3)
    if most_common:
        tonic = most_common[0][0]
        # Simple heuristic: check if major or minor
        # This is very basic - real scale detection is more complex
        return f"{tonic} (vermutlich)"
    
    return None


# ============================================================================
# APERTUS AI FEEDBACK (Enhanced with note data)
# ============================================================================

async def get_apertus_feedback(audio_features: Dict, jazz_analysis: Dict, note_analysis: Dict) -> Dict:
    """Nutzt Apertus AI für intelligentes Feedback (now with RAG knowledge)"""
    
    if not apertus_client:
        print("Apertus nicht verfügbar, fallback")
        return None
    
    try:
        # Get relevant jazz knowledge from RAG system
        print("🔍 Searching knowledge base...")
        kb = get_knowledge_base()
        knowledge_context = kb.get_context_for_analysis(audio_features, jazz_analysis)
        print(f"✅ Retrieved {len(knowledge_context)} chars of context")
        
        # Enhanced prompt with note information
        note_info = ""
NOTE DETECTION (Basic Pitch):
- Erkannte Noten: {note_analysis['total_notes']}
- Tonumfang: {note_analysis['pitch_range']['min_note']} bis {note_analysis['pitch_range']['max_note']}
- Häufigste Noten: {', '.join(note_analysis['most_common_notes'][:5])}
- Vermutete Tonart: {note_analysis.get('detected_scale', 'unbekannt')}
"""
        
        prompt = f"""Du bist ein erfahrener Jazz-Lehrer mit 30 Jahren Unterrichtserfahrung. Analysiere diese Jazz-Improvisation und gib konstruktives, spezifisches Feedback.

AUDIO-DATEN (Librosa-Analyse):
- Dauer: {audio_features['duration']:.1f} Sekunden
- Tempo: {audio_features['tempo']:.1f} BPM
- Tempo-Stabilität: {audio_features['tempo_stability']:.1%}
- Noten-Dichte: {audio_features['note_density']:.2f} Noten/Sekunde
- Rhythmische Komplexität: {audio_features['rhythm_complexity']:.1f}/10
- Dynamik-Range: {audio_features['dynamics']['dynamic_range']:.1f}x
- Spektraler Centroid: {audio_features['spectral']['centroid_mean']:.0f} Hz

{note_info}

JAZZ-KONTEXT:
- Tempo-Kategorie: {jazz_analysis['tempo_category']}
- Referenz: {jazz_analysis['tempo_reference']}
RELEVANTE JAZZ-THEORIE (aus Knowledge Base):
{knowledge_context}

---
- Rhythmische Bewertung: {jazz_analysis['rhythm_assessment']}
- Phrasierungs-Dichte: {jazz_analysis['density_assessment']}
- Swing-Feel: {jazz_analysis['swing_feel']}
- Ähnliche Künstler: {', '.join(jazz_analysis['similar_artists'])}

Gib detailliertes Feedback in 4 Kategorien (je 1-10 Punkte):
1. Rhythmus & Timing - Analysiere Tempo-Stabilität, Swing-Feel, rhythmische Komplexität
2. Harmonie - Bewerte basierend auf spektralen Eigenschaften, Note Detection und Jazz-Kontext
3. Melodie & Phrasierung - Beurteile Noten-Dichte, Phrasenlänge, melodische Entwicklung, Tonumfang
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
# BACKGROUND PROCESSING (Enhanced with Basic Pitch)
# ============================================================================

def process_audio_in_background(analysis_id: str, tmp_path: str):
    """Background task with Librosa + Basic Pitch + Apertus (Memory Optimized)"""
    import asyncio
    import gc
    
    try:
        # Step 1: Librosa
        analysis_results[analysis_id] = {"status": "processing", "stage": "librosa"}
        audio_features = analyze_audio_file(tmp_path)
        jazz_analysis = analyze_jazz_patterns(audio_features)
        
        # Free memory after Librosa
        gc.collect()
        
        # Step 2: Basic Pitch Note Detection (only for short files to avoid timeout/memory)
        note_analysis = None
        if False:  # DISABLED - causes OOM crashes on 512MB RAM
            analysis_results[analysis_id] = {"status": "processing", "stage": "notes"}
            note_analysis = analyze_notes_with_basic_pitch(tmp_path)
            # Memory already freed inside function
        else:
            print(f"⏭️  Skipping Basic Pitch for {audio_features['duration']:.1f}s file (too long)")
        
        # Free memory before AI call
        gc.collect()
        
        # Step 3: Apertus AI
        analysis_results[analysis_id] = {"status": "processing", "stage": "ai"}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        feedback = loop.run_until_complete(
            get_apertus_feedback(audio_features, jazz_analysis, note_analysis)
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
            "note_analysis": note_analysis,  # Can be None for long files
            "feedback": feedback,
            "ai_generated": apertus_client is not None
        }
        
        analysis_results[analysis_id] = {
            "status": "completed",
            "result": result
        }
        
        # Final memory cleanup
        gc.collect()
        print("✅ Analysis complete, memory cleaned up")
        
    except Exception as e:
        print(f"Error in background processing: {e}")
        import traceback
        traceback.print_exc()
        analysis_results[analysis_id] = {
            "status": "error",
            "error": str(e)
        }
        # Clean up on error
        gc.collect()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        # Final cleanup
        gc.collect()


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
        "ai_enabled": apertus_client is not None,
        "note_detection": "Basic Pitch"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
