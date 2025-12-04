# Jazz Improvisation Feedback Platform - Backend Implementation
# FastAPI + Librosa + Basic Pitch + Claude API

"""
INSTALLATION:
pip install fastapi uvicorn librosa numpy scipy basic-pitch anthropic python-multipart

DEPLOYMENT:
- Render.com (kostenlos für Start)
- Railway.app (kostenlos für Start)
- Fly.io (kostenlos für Start)

JAZZ DATASETS INTEGRATION:
1. Weimar Jazz Database: https://jazzomat.hfm-weimar.de/
2. Download transkribierte Solos (MIDI + Annotations)
3. Nutze für Pattern-Matching und Referenz-Vergleiche
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from scipy import signal
import tempfile
import os
from typing import Dict, List, Optional
import json

# Für Production: import anthropic

app = FastAPI(title="Jazz Feedback API")

# CORS für Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Production: specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# AUDIO ANALYSIS FUNCTIONS (Librosa-basiert)
# ============================================================================

def analyze_audio_file(audio_path: str) -> Dict:
    """
    Echte Audio-Analyse mit Librosa
    """
    # Audio laden
    y, sr = librosa.load(audio_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. TEMPO & BEAT DETECTION
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    tempo_stability = calculate_tempo_stability(beat_times)
    
    # 2. PITCH & MELODY ANALYSIS
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_sequence = extract_pitch_sequence(pitches, magnitudes)
    
    # 3. HARMONIC ANALYSIS
    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    
    # 4. RHYTHM COMPLEXITY
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    rhythm_complexity = calculate_rhythm_complexity(onsets)
    
    # 5. DYNAMICS (RMS Energy)
    rms = librosa.feature.rms(y=y)[0]
    dynamic_range = float(np.max(rms) / (np.mean(rms) + 1e-6))
    dynamic_variance = float(np.std(rms))
    
    # 6. SPECTRAL FEATURES
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    # 7. MFCC (Timbre)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
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
    """
    Berechnet wie stabil das Tempo ist (0-1)
    """
    if len(beat_times) < 3:
        return 0.5
    
    intervals = np.diff(beat_times)
    stability = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-6))
    return float(np.clip(stability, 0, 1))


def extract_pitch_sequence(pitches: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
    """
    Extrahiert dominante Tonhöhen über Zeit
    """
    pitch_sequence = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_sequence.append(pitch)
    return np.array(pitch_sequence)


def calculate_rhythm_complexity(onsets: np.ndarray) -> float:
    """
    Rhythmische Komplexität basierend auf Onset-Intervall-Variabilität
    """
    if len(onsets) < 3:
        return 1.0
    
    intervals = np.diff(onsets)
    # Normalisierte Standardabweichung
    complexity = np.std(intervals) / (np.mean(intervals) + 1e-6)
    # Skaliere auf 0-10
    return float(np.clip(complexity * 5, 0, 10))


# ============================================================================
# JAZZ-SPECIFIC ANALYSIS (Mit Weimar Jazz Database Pattern-Matching)
# ============================================================================

# Simulierte Jazz-Patterns aus Weimar Database
# In Production: Diese aus echter Database laden
JAZZ_PATTERNS = {
    "bebop_licks": [
        {"name": "Charlie Parker Lick", "intervals": [2, 2, 1, 2, 2, 2], "tempo_range": (140, 180)},
        {"name": "Clifford Brown Pattern", "intervals": [2, 1, 2, 2, 1, 2], "tempo_range": (130, 170)},
    ],
    "ii_v_i_patterns": [
        {"progression": ["Dm7", "G7", "Cmaj7"], "common_rhythms": [0.5, 0.25, 0.25, 1.0]}
    ],
    "swing_ratios": {
        "classic_swing": 0.67,  # 2:1 Verhältnis
        "modern_swing": 0.60,
        "straight_eighth": 0.50
    }
}


def analyze_jazz_patterns(audio_features: Dict) -> Dict:
    """
    Vergleicht Aufnahme mit typischen Jazz-Patterns
    """
    tempo = audio_features["tempo"]
    rhythm_complexity = audio_features["rhythm_complexity"]
    note_density = audio_features["note_density"]
    
    # Tempo-Klassifikation
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
    
    # Rhythmische Komplexität bewerten
    if rhythm_complexity < 3:
        rhythm_assessment = "Einfache, klare Phrasierung"
    elif 3 <= rhythm_complexity < 6:
        rhythm_assessment = "Moderate Komplexität, gut balanciert"
    else:
        rhythm_assessment = "Sehr komplex, evtl. zu hektisch"
    
    # Noten-Dichte bewerten
    if note_density < 2:
        density_assessment = "Sparsame Phrasierung (Monk-Stil)"
    elif 2 <= note_density < 4:
        density_assessment = "Ausgewogene Dichte (Mainstream Jazz)"
    else:
        density_assessment = "Sehr dichte Lines (Bebop/Coltrane-Stil)"
    
    return {
        "tempo_category": tempo_category,
        "tempo_reference": reference,
        "rhythm_assessment": rhythm_assessment,
        "density_assessment": density_assessment,
        "swing_feel": estimate_swing_feel(audio_features),
        "similar_artists": get_similar_artists(audio_features)
    }


def estimate_swing_feel(features: Dict) -> str:
    """
    Schätzt Swing-Feel basierend auf rhythmischen Features
    """
    complexity = features["rhythm_complexity"]
    if complexity < 3:
        return "Straight/Even Eighths"
    elif 3 <= complexity < 6:
        return "Classic Swing Feel (2:1)"
    else:
        return "Complex/Modern Swing"


def get_similar_artists(features: Dict) -> List[str]:
    """
    Findet ähnliche Jazz-Künstler basierend auf Features
    """
    tempo = features["tempo"]
    density = features["note_density"]
    
    artists = []
    
    if tempo < 100 and density < 2:
        artists.extend(["Bill Evans", "Keith Jarrett"])
    elif 140 <= tempo < 200 and density > 3:
        artists.extend(["Charlie Parker", "Dizzy Gillespie"])
    elif density > 4:
        artists.extend(["John Coltrane", "Michael Brecker"])
    else:
        artists.extend(["Miles Davis", "Dexter Gordon"])
    
    return artists


# ============================================================================
# CLAUDE API INTEGRATION
# ============================================================================

async def get_claude_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    """
    Nutzt Claude für intelligentes, pädagogisches Feedback
    """
    # Für Demo: Simuliertes Claude-Feedback
    # In Production: Echte Claude API nutzen
    
    prompt = f"""Du bist ein erfahrener Jazz-Lehrer mit 30 Jahren Erfahrung. 

AUDIO-ANALYSE DATEN:
- Tempo: {audio_features['tempo']:.1f} BPM ({jazz_analysis['tempo_category']})
- Tempo-Stabilität: {audio_features['tempo_stability']:.2f}
- Noten-Dichte: {audio_features['note_density']:.2f} Noten/Sekunde
- Rhythmische Komplexität: {audio_features['rhythm_complexity']:.1f}/10
- Dynamik-Range: {audio_features['dynamics']['dynamic_range']:.2f}x
- Spektraler Centroid: {audio_features['spectral']['centroid_mean']:.0f} Hz

JAZZ-KONTEXT:
- Stil: {jazz_analysis['tempo_category']}
- Referenz: {jazz_analysis['tempo_reference']}
- Rhythmik: {jazz_analysis['rhythm_assessment']}
- Dichte: {jazz_analysis['density_assessment']}
- Swing-Feel: {jazz_analysis['swing_feel']}
- Ähnliche Künstler: {', '.join(jazz_analysis['similar_artists'])}

Gib eine ehrliche, konstruktive Bewertung (1-10) mit SPEZIFISCHEM Feedback zu:
1. Rhythmus & Timing
2. Harmonische Auswahl
3. Melodische Entwicklung
4. Artikulation & Dynamik

Antworte NUR mit JSON (kein Markdown):
{{"rhythm": {{"score": 7.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "harmony": {{"score": 8.0, "feedback": "...", "tips": ["...", "...", "..."]}}, "melody": {{"score": 6.5, "feedback": "...", "tips": ["...", "...", "..."]}}, "articulation": {{"score": 7.0, "feedback": "...", "tips": ["...", "...", "..."]}}}}"""
    
    # PRODUCTION CODE (auskommentiert für Demo):
    # client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # message = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=1500,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # response = message.content[0].text
    # return json.loads(response)
    
    # Demo Fallback:
    return generate_rule_based_feedback(audio_features, jazz_analysis)


def generate_rule_based_feedback(audio_features: Dict, jazz_analysis: Dict) -> Dict:
    """
    Regel-basiertes Feedback als Fallback
    """
    tempo_stability = audio_features["tempo_stability"]
    rhythm_complexity = audio_features["rhythm_complexity"]
    dynamic_range = audio_features["dynamics"]["dynamic_range"]
    note_density = audio_features["note_density"]
    
    # Rhythmus Score
    rhythm_score = (tempo_stability * 5) + (5 if 3 <= rhythm_complexity <= 6 else 3)
    rhythm_feedback = f"Tempo-Stabilität bei {tempo_stability:.0%}. {jazz_analysis['rhythm_assessment']}. {jazz_analysis['tempo_reference']}."
    
    # Dynamik Score
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
            "feedback": f"Harmonische Inhalte erkannt. Spektraler Centroid bei {audio_features['spectral']['centroid_mean']:.0f} Hz deutet auf mittlere Tonlage hin.",
            "tips": [
                "Experimentiere mit ii-V-I Progressionen",
                "Nutze chromatische Approach-Töne",
                "Studiere Bebop-Scales für mehr harmonische Freiheit"
            ]
        },
        "melody": {
            "score": 6.5 if 2 <= note_density <= 4 else 5.5,
            "feedback": f"{jazz_analysis['density_assessment']}. Noten-Dichte: {note_density:.2f}/Sekunde.",
            "tips": [
                f"Variiere Phrasenlängen - aktuell eher {'dicht' if note_density > 3 else 'sparsam'}",
                "Nutze mehr Pausen zwischen Phrasen",
                f"Studiere {', '.join(jazz_analysis['similar_artists'][:2])} für melodische Ideen"
            ]
        },
        "articulation": {
            "score": round(articulation_score, 1),
            "feedback": articulation_feedback,
            "tips": [
                "Arbeite an extremeren dynamischen Kontrasten",
                "Nutze Akzente für rhythmische Betonung",
                "Experimentiere mit Ghost Notes für mehr Groove"
            ]
        }
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Hauptendpoint: Audio-Datei hochladen und Analyse erhalten
    """
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Nur Audio-Dateien erlaubt")
    
    # Temporäre Datei speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # 1. Audio-Analyse mit Librosa
        audio_features = analyze_audio_file(tmp_path)
        
        # 2. Jazz-spezifische Analyse
        jazz_analysis = analyze_jazz_patterns(audio_features)
        
        # 3. Claude Feedback (oder regelbasiert)
        feedback = await get_claude_feedback(audio_features, jazz_analysis)
        
        # Gesamtscore berechnen
        overall_score = (
            feedback["rhythm"]["score"] +
            feedback["harmony"]["score"] +
            feedback["melody"]["score"] +
            feedback["articulation"]["score"]
        ) / 4
        
        return {
            "overall_score": round(overall_score, 1),
            "audio_features": audio_features,
            "jazz_analysis": jazz_analysis,
            "feedback": feedback
        }
        
    finally:
        # Cleanup
        os.unlink(tmp_path)


@app.get("/")
async def root():
    return {
        "service": "Jazz Improvisation Feedback API",
        "version": "1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
DEPLOYMENT GUIDE:

1. RENDER.COM (Empfohlen):
   - Neuer Web Service
   - GitHub Repo verbinden
   - Build Command: pip install -r requirements.txt
   - Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   - Free Tier: 750 Stunden/Monat

2. RAILWAY.APP:
   - GitHub Repo verbinden
   - Automatische Erkennung
   - Free Tier: $5 Credit/Monat

3. REQUIREMENTS.TXT:
fastapi==0.104.1
uvicorn[standard]==0.24.0
librosa==0.10.1
numpy==1.24.3
scipy==1.11.3
python-multipart==0.0.6
anthropic==0.7.0

4. ENVIRONMENT VARIABLES:
   - ANTHROPIC_API_KEY=your_key_here

5. FRONTEND INTEGRATION:
   Ändere im React-Frontend die URL:
   fetch('https://your-backend.onrender.com/analyze', ...)
"""
