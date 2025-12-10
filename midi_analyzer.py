"""
MIDI Analysis Module - IMPROVED VERSION v3
- Better chord separation (based on time gaps between note groups)
- C Major default for jazz (common jazz key)
- Correct enharmonic spelling (b9 not #8, b5 not #4)
- Better ii-V-I detection
"""

import mido
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


def midi_to_note_name(midi_number: int, prefer_flat: bool = False) -> str:
    """Convert MIDI note number to note name with optional flat preference"""
    if prefer_flat:
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    else:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"


def midi_to_note_class(midi_number: int, prefer_flat: bool = False) -> str:
    """Convert MIDI note number to note class without octave"""
    if prefer_flat:
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    else:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[midi_number % 12]


def analyze_midi_file(midi_path: str) -> Dict:
    """
    Analyze MIDI file and extract all musical information
    IMPROVED: Better chord separation and key detection
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
        
        # IMPROVED: Detect chords with better separation
        # Use larger gap threshold (0.3s = approximately a beat at 120bpm)
        chords = detect_chords_improved(notes, onset_window=0.08, gap_threshold=0.25)
        print(f"   Found {len(chords)} chord groups")
        
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
        
        # IMPROVED: Detect key - with jazz bias towards common keys
        detected_key = detect_key_jazz_aware(notes, chord_analysis)
        
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
        
        print(f"✅ MIDI analyzed: {len(notes)} notes, {len(chords)} chords, key: {detected_key}")
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
    """Extract all notes with CORRECT timing in seconds"""
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
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes


def detect_chords_improved(notes: List[Dict], onset_window: float = 0.08, gap_threshold: float = 0.25) -> List[List[Dict]]:
    """
    IMPROVED chord detection:
    1. Group notes that start within onset_window (simultaneous)
    2. Separate chord groups when there's a gap > gap_threshold
    """
    if not notes:
        return []
    
    # First: group notes by similar onset time
    onset_groups = []
    current_group = [notes[0]]
    
    for note in notes[1:]:
        # Check if this note starts close to the first note in current group
        if abs(note['start'] - current_group[0]['start']) <= onset_window:
            current_group.append(note)
        else:
            if current_group:
                onset_groups.append(current_group)
            current_group = [note]
    
    if current_group:
        onset_groups.append(current_group)
    
    # Second: merge onset groups that are close together, separate those with gaps
    if not onset_groups:
        return []
    
    chords = []
    current_chord_notes = onset_groups[0]
    current_chord_time = min(n['start'] for n in current_chord_notes)
    
    for group in onset_groups[1:]:
        group_time = min(n['start'] for n in group)
        
        # If there's a significant gap, this is a new chord
        if group_time - current_chord_time > gap_threshold:
            # Save previous chord if it has multiple notes
            if len(current_chord_notes) >= 2:
                chords.append(current_chord_notes)
            current_chord_notes = group
            current_chord_time = group_time
        else:
            # Merge with current chord (could be arpeggiated)
            # Only merge if it doesn't make the chord too big
            if len(current_chord_notes) + len(group) <= 8:
                current_chord_notes.extend(group)
            else:
                if len(current_chord_notes) >= 2:
                    chords.append(current_chord_notes)
                current_chord_notes = group
                current_chord_time = group_time
    
    # Don't forget last chord
    if len(current_chord_notes) >= 2:
        chords.append(current_chord_notes)
    
    return chords


def analyze_chord(chord: List[Dict]) -> Dict:
    """Analyze a chord and identify its type with jazz chord symbols"""
    # Sort notes by pitch
    sorted_notes = sorted(chord, key=lambda x: x['pitch'])
    pitches = [n['pitch'] for n in sorted_notes]
    note_names = [n['note_name'] for n in sorted_notes]
    
    # Get unique pitch classes
    pitch_classes = list(set(p % 12 for p in pitches))
    
    # Find root and chord type
    bass_note = pitches[0]
    bass_class = bass_note % 12
    bass_name = midi_to_note_class(bass_note, prefer_flat=True)
    
    # Identify jazz chord
    chord_type, root, root_pc = identify_jazz_chord_v2(pitch_classes, bass_class)
    
    # Get proper note names for the chord
    note_classes = [midi_to_note_class(p, prefer_flat=True) for p in pitches]
    
    # Create chord symbol
    if root:
        symbol = f"{root}{chord_type}"
        if bass_name != root:
            symbol = f"{symbol}/{bass_name}"
    else:
        symbol = f"{bass_name}{chord_type}"
    
    return {
        'symbol': symbol,
        'root': root or bass_name,
        'type': chord_type,
        'bass': bass_name,
        'pitches': pitches,
        'notes': note_classes,
        'start_time': min(n['start'] for n in chord),
        'num_notes': len(chord)
    }


def identify_jazz_chord_v2(pitch_classes: List[int], bass_pc: int) -> Tuple[str, str, int]:
    """
    Identify jazz chord with better voicing recognition
    Returns: (chord_type, root_name, root_pitch_class)
    """
    pcs = set(pitch_classes)
    
    # Note names with flats for jazz (more common in jazz)
    note_names_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    # Jazz chord templates: (intervals from root, chord_type)
    # Intervals: 0=root, 1=b9, 2=9, 3=m3/#9, 4=M3, 5=11, 6=b5/#11, 7=5, 8=b13/#5, 9=13/6, 10=b7, 11=maj7
    chord_templates = [
        # 7th chords (most common in jazz)
        ({0, 4, 7, 11}, "maj7"),      # C E G B
        ({0, 4, 7, 10}, "7"),          # C E G Bb (dominant)
        ({0, 3, 7, 10}, "m7"),         # C Eb G Bb
        ({0, 3, 6, 10}, "m7b5"),       # C Eb Gb Bb (half-dim)
        ({0, 3, 6, 9}, "dim7"),        # C Eb Gb A (fully dim)
        
        # Extended dominant chords
        ({0, 4, 7, 10, 2}, "9"),       # C E G Bb D
        ({0, 4, 7, 10, 1}, "7b9"),     # C E G Bb Db
        ({0, 4, 7, 10, 3}, "7#9"),     # C E G Bb D#
        ({0, 4, 6, 10}, "7b5"),        # C E Gb Bb
        ({0, 4, 8, 10}, "7#5"),        # C E G# Bb
        ({0, 4, 6, 10, 1}, "7b5b9"),   # C E Gb Bb Db
        ({0, 4, 7, 10, 2, 9}, "13"),   # C E G Bb D A
        
        # Extended minor chords
        ({0, 3, 7, 10, 2}, "m9"),      # C Eb G Bb D
        ({0, 3, 7, 11}, "mMaj7"),      # C Eb G B
        
        # Extended major chords
        ({0, 4, 7, 11, 2}, "maj9"),    # C E G B D
        ({0, 4, 7, 11, 6}, "maj7#11"), # C E G B F#
        
        # Rootless voicings (common in jazz piano)
        # ii7 rootless: 3-5-7-9 = F A C E for Dm7
        ({0, 4, 7, 11}, "maj7"),  # Could be rootless dom7 (3-5-7-9)
        
        # Sus chords
        ({0, 5, 7, 10}, "7sus4"),      # C F G Bb
        ({0, 2, 7, 10}, "7sus2"),      # C D G Bb
        ({0, 5, 7}, "sus4"),           # C F G
        ({0, 2, 7}, "sus2"),           # C D G
        
        # Triads (less common alone in jazz)
        ({0, 4, 7}, ""),               # C E G (major)
        ({0, 3, 7}, "m"),              # C Eb G (minor)
        ({0, 3, 6}, "dim"),            # C Eb Gb
        ({0, 4, 8}, "aug"),            # C E G#
        
        # 6th chords
        ({0, 4, 7, 9}, "6"),           # C E G A
        ({0, 3, 7, 9}, "m6"),          # C Eb G A
    ]
    
    best_match = ("?", None, bass_pc)
    best_match_size = 0
    
    # Try each possible root
    for root_pc in range(12):
        # Transpose intervals to this root
        intervals = set((pc - root_pc) % 12 for pc in pcs)
        
        # Check against templates
        for template_intervals, chord_type in chord_templates:
            # Check how many notes match
            matching = len(intervals & template_intervals)
            
            # Exact match or close match (allowing extensions)
            if template_intervals.issubset(intervals) or intervals.issubset(template_intervals):
                if matching > best_match_size:
                    best_match_size = matching
                    root_name = note_names_flat[root_pc]
                    best_match = (chord_type, root_name, root_pc)
            # Good partial match
            elif matching >= 3 and matching >= len(template_intervals) - 1:
                if matching > best_match_size:
                    best_match_size = matching
                    root_name = note_names_flat[root_pc]
                    best_match = (chord_type, root_name, root_pc)
    
    return best_match


def detect_key_jazz_aware(notes: List[Dict], chords: List[Dict]) -> str:
    """
    Detect key with jazz awareness:
    - Prefer common jazz keys (C, F, Bb, Eb, G, D)
    - Use chord analysis to help
    - Don't get confused by chromatic alterations
    """
    if not notes:
        return "C Major"  # Default for jazz
    
    # Count note classes (but ignore rare chromatic notes)
    note_classes = [n['note_class'] for n in notes]
    counter = Counter(note_classes)
    total = sum(counter.values())
    
    # Filter out notes that appear less than 5% (likely chromatic passing tones)
    filtered_notes = {note: count for note, count in counter.items() 
                     if count / total >= 0.05}
    
    # Common jazz keys (in order of preference)
    common_jazz_keys = ['C', 'F', 'Bb', 'G', 'Eb', 'D', 'Ab', 'A', 'E', 'Db', 'B', 'Gb']
    
    # Key profiles (Krumhansl-Kessler)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    note_order_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_order_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    # Calculate correlation with each key
    key_scores = []
    
    for root_idx in range(12):
        root_sharp = note_order_sharp[root_idx]
        root_flat = note_order_flat[root_idx]
        
        # Create distribution for this root
        dist = []
        for i in range(12):
            note_sharp = note_order_sharp[(root_idx + i) % 12]
            note_flat = note_order_flat[(root_idx + i) % 12]
            count = counter.get(note_sharp, 0) + counter.get(note_flat, 0)
            if note_sharp != note_flat:
                count = max(counter.get(note_sharp, 0), counter.get(note_flat, 0))
            dist.append(count / total if total > 0 else 0)
        
        # Correlate with major
        try:
            major_corr = float(np.corrcoef(dist, major_profile)[0, 1])
        except:
            major_corr = 0
        
        # Correlate with minor
        try:
            minor_corr = float(np.corrcoef(dist, minor_profile)[0, 1])
        except:
            minor_corr = 0
        
        # Boost score for common jazz keys
        root_name = root_flat if root_flat in common_jazz_keys else root_sharp
        jazz_boost = 0.05 if root_name in common_jazz_keys[:6] else 0
        
        key_scores.append((root_name, "Major", major_corr + jazz_boost))
        key_scores.append((root_name, "Minor", minor_corr + jazz_boost))
    
    # Sort by score
    key_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Take best result
    best_root, best_mode, best_score = key_scores[0]
    
    # If chord analysis suggests ii-V-I, use that to determine key
    if chords and len(chords) >= 3:
        chord_types = [c.get('type', '') for c in chords[:4]]
        chord_roots = [c.get('root', '') for c in chords[:4]]
        
        # Look for ii-V-I pattern (m7 -> 7 -> maj7)
        for i in range(len(chord_types) - 2):
            if ('m7' in chord_types[i] and 'm7b5' not in chord_types[i] and
                chord_types[i+1] == '7' and 
                'maj7' in chord_types[i+2]):
                # Found ii-V-I! The I chord root is the key
                key_root = chord_roots[i+2]
                print(f"   ii-V-I detected! Key: {key_root} Major")
                return f"{key_root} Major"
    
    return f"{best_root} {best_mode}"


def detect_progression(chords: List[Dict], key: str) -> Dict:
    """Detect chord progression and analyze in context of key"""
    if not chords:
        return {"progression": [], "analysis": "No chords detected", "type": "None"}
    
    # Extract chord info
    symbols = [c['symbol'] for c in chords]
    types = [c.get('type', '') for c in chords]
    roots = [c.get('root', '') for c in chords]
    
    # Detect ii-V-I
    progression_type = "Custom"
    
    for i in range(len(types) - 2):
        # ii-V-I: m7 -> 7 -> maj7
        if ('m7' in types[i] and 'm7b5' not in types[i] and
            types[i+1] in ['7', '7b9', '7#9', '7b5', '7#5', '9', '13'] and
            'maj7' in types[i+2]):
            progression_type = "ii-V-I"
            break
        # ii-V-i (minor): m7b5 -> 7 -> m7
        if ('m7b5' in types[i] and
            types[i+1] in ['7', '7b9', '7#9', '7b5'] and
            'm7' in types[i+2] and 'maj' not in types[i+2]):
            progression_type = "ii-V-i (minor)"
            break
    
    # Check for blues
    if len([t for t in types if t == '7']) >= 3:
        progression_type = "Blues"
    
    # Create Roman numeral analysis
    roman_numerals = []
    if "Major" in key or "Minor" in key:
        key_root = key.split()[0]
        is_major = "Major" in key
        for chord in chords:
            roman = get_roman_numeral(chord.get('root', ''), key_root, chord.get('type', ''), is_major)
            roman_numerals.append(roman)
    
    analysis_text = f"{progression_type} progression in {key}" if progression_type != "Custom" else f"Chord progression in {key}"
    
    return {
        "symbols": symbols,
        "type": progression_type,
        "roman_numerals": roman_numerals,
        "analysis": analysis_text
    }


def get_roman_numeral(chord_root: str, key_root: str, chord_type: str, is_major: bool) -> str:
    """Convert chord to Roman numeral in given key"""
    # Handle both sharp and flat note names
    note_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10,
        'Bb': 10, 'B': 11, 'Cb': 11
    }
    
    key_pc = note_to_pc.get(key_root, 0)
    chord_pc = note_to_pc.get(chord_root, 0)
    
    degree = (chord_pc - key_pc) % 12
    
    # Roman numerals for scale degrees
    if is_major:
        numerals = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']
    else:
        numerals = ['i', 'bII', 'ii', 'bIII', 'III', 'iv', '#iv', 'v', 'bVI', 'VI', 'bVII', 'VII']
    
    numeral = numerals[degree]
    
    # Adjust case based on chord quality
    if chord_type:
        if 'm7' in chord_type and 'maj' not in chord_type.lower():
            numeral = numeral.lower()
        elif chord_type in ['', 'maj7', '6', 'maj9']:
            numeral = numeral.upper()
    
    # Add type suffix
    if chord_type and chord_type not in ['', 'm']:
        type_suffix = chord_type
        if type_suffix.startswith('m') and 'maj' not in type_suffix:
            type_suffix = type_suffix[1:]  # Remove 'm' since lowercase numeral indicates minor
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
        "precision_score": min(1.0, precision_score),
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
        
        pitches1 = set(chord1.get('pitches', []))
        pitches2 = set(chord2.get('pitches', []))
        
        common_tones = pitches1 & pitches2
        
        # Calculate movements
        movements = []
        for p1 in sorted(chord1.get('pitches', [])):
            if chord2.get('pitches'):
                closest = min(chord2['pitches'], key=lambda p2: abs(p2 - p1))
                movement = closest - p1
                movements.append(movement)
        
        voice_leading.append({
            'from': chord1.get('symbol', '?'),
            'to': chord2.get('symbol', '?'),
            'common_tones': len(common_tones),
            'movements': movements,
            'smooth': all(abs(m) <= 2 for m in movements) if movements else True
        })
    
    return voice_leading
