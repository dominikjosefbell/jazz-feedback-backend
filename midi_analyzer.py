"""
MIDI Analysis Module - IMPROVED VERSION
Correct timing calculation + chord recognition
"""

import mido
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"


def midi_to_note_class(midi_number: int) -> str:
    """Convert MIDI note number to note class without octave (e.g., 60 -> 'C')"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[midi_number % 12]


def analyze_midi_file(midi_path: str) -> Dict:
    """
    Analyze MIDI file and extract all musical information
    FIXED: Correct timing calculation using ticks_per_beat
    """
    try:
        print("🎹 Analyzing MIDI file...")
        
        mid = mido.MidiFile(midi_path)
        
        # Get ticks per beat for timing calculation
        ticks_per_beat = mid.ticks_per_beat
        print(f"   Ticks per beat: {ticks_per_beat}")
        
        # Extract tempo (microseconds per beat)
        tempo = 500000  # Default: 120 BPM
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        tempo_bpm = mido.tempo2bpm(tempo)
        print(f"   Tempo: {tempo_bpm:.1f} BPM")
        
        # Calculate seconds per tick
        seconds_per_tick = tempo / (ticks_per_beat * 1000000)
        
        # Extract all notes with CORRECT timing
        notes = extract_notes_with_timing(mid, seconds_per_tick)
        
        if not notes:
            return {
                "total_notes": 0,
                "error": "No notes found in MIDI file"
            }
        
        print(f"   Found {len(notes)} notes")
        
        # Calculate actual duration
        duration = max(n['end'] for n in notes) if notes else 0
        print(f"   Duration: {duration:.2f} seconds")
        
        # Detect chords (notes played within 50ms)
        chords = detect_chords(notes, time_window=0.05)
        print(f"   Found {len(chords)} chords")
        
        # Analyze each chord and identify type
        chord_analysis = []
        for chord in chords:
            chord_info = analyze_chord(chord)
            chord_analysis.append(chord_info)
            print(f"   Chord: {chord_info['symbol']} at {chord_info['start_time']:.2f}s")
        
        # Most common notes
        note_names = [n['note_name'] for n in notes]
        note_counter = Counter(note_names)
        most_common = note_counter.most_common(5)
        
        # Pitch range
        pitches = [n['pitch'] for n in notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        
        # Detect key
        detected_key = detect_key_from_notes(notes)
        
        # Analyze timing precision
        timing_analysis = analyze_timing(notes)
        
        # Analyze dynamics
        dynamics_analysis = analyze_dynamics(notes)
        
        # Detect chord progression
        progression = detect_progression(chord_analysis, detected_key)
        
        result = {
            "total_notes": len(notes),
            "notes": notes[:100],  # First 100 for display
            "pitch_range": {
                "min": min_pitch,
                "max": max_pitch,
                "min_note": midi_to_note_name(min_pitch),
                "max_note": midi_to_note_name(max_pitch)
            },
            "most_common_notes": [note for note, count in most_common],
            "detected_scale": detected_key,
            "tempo_bpm": tempo_bpm,
            "duration": duration,
            "chords": chord_analysis,
            "chord_symbols": [c['symbol'] for c in chord_analysis],
            "progression": progression,
            "timing": timing_analysis,
            "dynamics": dynamics_analysis
        }
        
        print(f"✅ MIDI analyzed: {len(notes)} notes, {len(chords)} chords")
        return result
        
    except Exception as e:
        print(f"❌ MIDI analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "total_notes": 0,
            "error": str(e)
        }


def extract_notes_with_timing(mid: mido.MidiFile, seconds_per_tick: float) -> List[Dict]:
    """
    Extract all notes with CORRECT timing in seconds
    """
    notes = []
    
    for track in mid.tracks:
        current_time_ticks = 0
        active_notes = {}  # pitch -> {start_ticks, velocity}
        
        for msg in track:
            current_time_ticks += msg.time
            current_time_seconds = current_time_ticks * seconds_per_tick
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'start_ticks': current_time_ticks,
                    'start_seconds': current_time_seconds,
                    'velocity': msg.velocity
                }
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    note_data = active_notes[msg.note]
                    notes.append({
                        'pitch': msg.note,
                        'note_name': midi_to_note_name(msg.note),
                        'note_class': midi_to_note_class(msg.note),
                        'start': note_data['start_seconds'],
                        'end': current_time_seconds,
                        'duration': current_time_seconds - note_data['start_seconds'],
                        'velocity': note_data['velocity']
                    })
                    del active_notes[msg.note]
    
    # Sort by start time
    notes.sort(key=lambda x: x['start'])
    return notes


def detect_chords(notes: List[Dict], time_window: float = 0.05) -> List[List[Dict]]:
    """
    Detect chords (notes played within time_window seconds)
    """
    if not notes:
        return []
    
    chords = []
    current_chord = [notes[0]]
    current_time = notes[0]['start']
    
    for note in notes[1:]:
        if abs(note['start'] - current_time) <= time_window:
            current_chord.append(note)
        else:
            if len(current_chord) >= 2:
                chords.append(current_chord)
            elif len(current_chord) == 1:
                # Single notes can also be part of analysis
                pass
            current_chord = [note]
            current_time = note['start']
    
    # Don't forget last chord
    if len(current_chord) >= 2:
        chords.append(current_chord)
    
    return chords


def analyze_chord(chord: List[Dict]) -> Dict:
    """
    Analyze a chord and identify its type with jazz chord symbols
    """
    # Sort notes by pitch
    sorted_notes = sorted(chord, key=lambda x: x['pitch'])
    pitches = [n['pitch'] for n in sorted_notes]
    note_names = [n['note_name'] for n in sorted_notes]
    note_classes = [n['note_class'] for n in sorted_notes]
    
    # Find root (lowest note, but could be inversion)
    bass_note = pitches[0]
    bass_name = midi_to_note_class(bass_note)
    
    # Calculate intervals from bass
    intervals_from_bass = [(p - bass_note) % 12 for p in pitches]
    intervals_set = set(intervals_from_bass)
    
    # Try to identify chord
    chord_type, root = identify_jazz_chord(pitches, note_classes)
    
    # Create chord symbol
    if root:
        symbol = f"{root}{chord_type}"
        if bass_name != root:
            symbol = f"{symbol}/{bass_name}"
    else:
        symbol = f"{bass_name}?"
    
    return {
        'symbol': symbol,
        'root': root or bass_name,
        'type': chord_type,
        'bass': bass_name,
        'notes': note_names,
        'note_classes': list(set(note_classes)),
        'pitches': pitches,
        'intervals': intervals_from_bass,
        'start_time': sorted_notes[0]['start'],
        'velocity': sum(n['velocity'] for n in sorted_notes) / len(sorted_notes),
        'is_inversion': root != bass_name if root else False
    }


def identify_jazz_chord(pitches: List[int], note_classes: List[str]) -> Tuple[str, Optional[str]]:
    """
    Identify jazz chord type and root
    Returns: (chord_type, root_note)
    """
    # Get unique pitch classes (0-11)
    pitch_classes = sorted(set(p % 12 for p in pitches))
    
    # Common jazz chord templates (intervals from root)
    # Format: (intervals, chord_symbol)
    chord_templates = [
        # 7th chords
        ({0, 4, 7, 11}, "maj7"),      # Major 7
        ({0, 3, 7, 10}, "m7"),        # Minor 7
        ({0, 4, 7, 10}, "7"),         # Dominant 7
        ({0, 3, 6, 10}, "m7b5"),      # Half-diminished (ø)
        ({0, 3, 6, 9}, "dim7"),       # Diminished 7
        ({0, 4, 8, 11}, "maj7#5"),    # Augmented major 7
        ({0, 3, 7, 11}, "mMaj7"),     # Minor-major 7
        
        # 9th chords
        ({0, 4, 7, 10, 14}, "9"),     # Dominant 9
        ({0, 4, 7, 11, 14}, "maj9"),  # Major 9
        ({0, 3, 7, 10, 14}, "m9"),    # Minor 9
        
        # With tensions (3-5-7-9 voicing = no root!)
        ({0, 4, 7, 10}, "7"),         # 3-5-7-9 of dom7 (root omitted)
        ({0, 3, 6, 9}, "m7b5"),       # 3-5-7-9 of m7b5
        
        # Triads
        ({0, 4, 7}, ""),              # Major
        ({0, 3, 7}, "m"),             # Minor
        ({0, 3, 6}, "dim"),           # Diminished
        ({0, 4, 8}, "aug"),           # Augmented
        
        # Sus chords
        ({0, 5, 7}, "sus4"),
        ({0, 2, 7}, "sus2"),
        ({0, 5, 7, 10}, "7sus4"),
    ]
    
    # Try each possible root
    for root_pc in range(12):
        # Transpose intervals to this root
        intervals = set((pc - root_pc) % 12 for pc in pitch_classes)
        
        # Check against templates
        for template_intervals, chord_type in chord_templates:
            if intervals == template_intervals or template_intervals.issubset(intervals):
                root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root_pc]
                return chord_type, root_name
    
    # Fallback: use bass as root
    return "?", None


def detect_key_from_notes(notes: List[Dict]) -> str:
    """Detect key from note distribution"""
    if not notes:
        return "Unknown"
    
    # Count note classes
    note_classes = [n['note_class'] for n in notes]
    counter = Counter(note_classes)
    
    # Key profiles (Krumhansl-Kessler)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    note_order = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate correlation with each key
    best_key = "C"
    best_score = -1
    
    total = sum(counter.values())
    for root_idx, root in enumerate(note_order):
        # Create distribution for this root
        dist = []
        for i in range(12):
            note = note_order[(root_idx + i) % 12]
            dist.append(counter.get(note, 0) / total if total > 0 else 0)
        
        # Correlate with major
        major_corr = np.corrcoef(dist, major_profile)[0, 1] if len(dist) == 12 else 0
        if major_corr > best_score:
            best_score = major_corr
            best_key = f"{root} Major"
        
        # Correlate with minor
        minor_corr = np.corrcoef(dist, minor_profile)[0, 1] if len(dist) == 12 else 0
        if minor_corr > best_score:
            best_score = minor_corr
            best_key = f"{root} Minor"
    
    return best_key


def detect_progression(chords: List[Dict], key: str) -> Dict:
    """
    Detect chord progression and analyze in context of key
    """
    if not chords:
        return {"progression": [], "analysis": "No chords detected"}
    
    # Extract chord symbols
    symbols = [c['symbol'] for c in chords]
    
    # Common jazz progressions
    common_progressions = {
        "ii-V-I": ["m7", "7", "maj7"],
        "I-vi-ii-V": ["maj7", "m7", "m7", "7"],
        "iii-vi-ii-V": ["m7", "m7", "m7", "7"],
        "Blues": ["7", "7", "7", "7"],
    }
    
    # Try to identify progression type
    progression_type = "Custom"
    chord_types = [c['type'] for c in chords[:4]]
    
    for name, pattern in common_progressions.items():
        if chord_types[:len(pattern)] == pattern:
            progression_type = name
            break
    
    # Create Roman numeral analysis if key is known
    roman_numerals = []
    if "Major" in key or "Minor" in key:
        key_root = key.split()[0]
        for chord in chords:
            roman = get_roman_numeral(chord['root'], key_root, chord['type'], "Major" in key)
            roman_numerals.append(roman)
    
    return {
        "symbols": symbols,
        "type": progression_type,
        "roman_numerals": roman_numerals,
        "analysis": f"{progression_type} progression in {key}" if progression_type != "Custom" else f"Chord progression in {key}"
    }


def get_roman_numeral(chord_root: str, key_root: str, chord_type: str, is_major: bool) -> str:
    """Convert chord to Roman numeral in given key"""
    note_order = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    try:
        key_idx = note_order.index(key_root)
        chord_idx = note_order.index(chord_root)
    except ValueError:
        return "?"
    
    degree = (chord_idx - key_idx) % 12
    
    # Roman numerals for major scale degrees
    major_numerals = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']
    
    numeral = major_numerals[degree]
    
    # Lowercase for minor chords
    if 'm' in chord_type and 'maj' not in chord_type.lower():
        numeral = numeral.lower()
    
    # Add chord type
    if chord_type and chord_type not in ['', 'm']:
        type_suffix = chord_type.replace('m', '') if chord_type.startswith('m') else chord_type
        numeral = f"{numeral}{type_suffix}"
    
    return numeral


def analyze_timing(notes: List[Dict]) -> Dict:
    """Analyze timing precision"""
    if len(notes) < 2:
        return {"precision_score": 1.0, "mean_interval": 0, "std_interval": 0}
    
    onsets = [n['start'] for n in notes]
    intervals = np.diff(onsets)
    
    if len(intervals) == 0:
        return {"precision_score": 1.0, "mean_interval": 0, "std_interval": 0}
    
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    
    # Precision: lower std relative to mean = more precise
    if mean_interval > 0:
        precision_score = max(0, 1.0 - (std_interval / mean_interval))
    else:
        precision_score = 1.0
    
    return {
        "precision_score": precision_score,
        "mean_interval": mean_interval,
        "std_interval": std_interval
    }


def analyze_dynamics(notes: List[Dict]) -> Dict:
    """Analyze dynamics from velocity"""
    if not notes:
        return {"min": 0, "max": 0, "mean": 0, "range": 0, "std": 0}
    
    velocities = [n['velocity'] for n in notes]
    
    return {
        "min": min(velocities),
        "max": max(velocities),
        "mean": float(np.mean(velocities)),
        "range": max(velocities) - min(velocities),
        "std": float(np.std(velocities))
    }


def analyze_voice_leading(chords: List[Dict]) -> List[Dict]:
    """Analyze voice leading between consecutive chords"""
    if len(chords) < 2:
        return []
    
    voice_leading = []
    
    for i in range(len(chords) - 1):
        chord1 = chords[i]
        chord2 = chords[i + 1]
        
        pitches1 = set(chord1['pitches'])
        pitches2 = set(chord2['pitches'])
        
        common_tones = pitches1 & pitches2
        
        # Calculate movements
        movements = []
        for p1 in sorted(chord1['pitches']):
            closest = min(chord2['pitches'], key=lambda p2: abs(p2 - p1))
            movement = closest - p1
            movements.append(movement)
        
        voice_leading.append({
            'from': chord1['symbol'],
            'to': chord2['symbol'],
            'common_tones': len(common_tones),
            'movements': movements,
            'smooth': all(abs(m) <= 2 for m in movements),
            'parallel_fifths': check_parallel_fifths(chord1['pitches'], chord2['pitches'])
        })
    
    return voice_leading


def check_parallel_fifths(pitches1: List[int], pitches2: List[int]) -> bool:
    """Check for parallel fifths (voice leading issue)"""
    # Simplified check
    for i, p1a in enumerate(pitches1[:-1]):
        for p1b in pitches1[i+1:]:
            if (p1b - p1a) % 12 == 7:  # Perfect fifth
                # Check if both voices move in parallel
                if len(pitches2) > i + 1:
                    p2a = pitches2[i] if i < len(pitches2) else pitches2[0]
                    p2b = pitches2[i+1] if i+1 < len(pitches2) else pitches2[-1]
                    if (p2b - p2a) % 12 == 7:
                        return True
    return False
