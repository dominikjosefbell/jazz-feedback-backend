"""
MIDI Analysis Module - ROBUST VERSION
- Better error handling for edge cases
- Manual parsing fallback if mido fails
- Improved chord detection
"""

import struct
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


def midi_to_note_name(midi_number: int, prefer_flat: bool = False) -> str:
    """Convert MIDI note number to note name"""
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


def parse_midi_manually(filepath: str) -> Tuple[List[Dict], float, int]:
    """
    Parse MIDI file manually without mido library.
    Returns: (notes, tempo_bpm, ticks_per_beat)
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    if data[:4] != b'MThd':
        raise ValueError("Not a valid MIDI file")
    
    header_length = struct.unpack('>I', data[4:8])[0]
    num_tracks = struct.unpack('>H', data[10:12])[0]
    ticks_per_beat = struct.unpack('>H', data[12:14])[0]
    
    pos = 8 + header_length
    tempo = 500000  # default 120 BPM
    
    all_notes = []
    active_notes = {}  # pitch -> start_time, velocity
    
    for track_num in range(num_tracks):
        if pos >= len(data) or data[pos:pos+4] != b'MTrk':
            break
        
        track_length = struct.unpack('>I', data[pos+4:pos+8])[0]
        track_data = data[pos+8:pos+8+track_length]
        
        i = 0
        current_time = 0
        running_status = 0
        
        while i < len(track_data):
            try:
                # Read delta time
                delta = 0
                while i < len(track_data):
                    byte = track_data[i]
                    delta = (delta << 7) | (byte & 0x7F)
                    i += 1
                    if not (byte & 0x80):
                        break
                
                current_time += delta
                
                if i >= len(track_data):
                    break
                
                status = track_data[i]
                
                if status == 0xFF:  # Meta event
                    i += 1
                    if i >= len(track_data):
                        break
                    meta_type = track_data[i]
                    i += 1
                    
                    # Read length
                    length = 0
                    while i < len(track_data):
                        byte = track_data[i]
                        length = (length << 7) | (byte & 0x7F)
                        i += 1
                        if not (byte & 0x80):
                            break
                    
                    if meta_type == 0x51 and i + 2 < len(track_data):  # Tempo
                        tempo = (track_data[i] << 16) | (track_data[i+1] << 8) | track_data[i+2]
                    
                    i += length
                    
                elif status == 0xF0 or status == 0xF7:  # SysEx
                    i += 1
                    length = 0
                    while i < len(track_data):
                        byte = track_data[i]
                        length = (length << 7) | (byte & 0x7F)
                        i += 1
                        if not (byte & 0x80):
                            break
                    i += length
                    
                elif status & 0x80:  # MIDI event with status byte
                    running_status = status
                    i += 1
                    
                    event_type = (status >> 4) & 0x0F
                    
                    if event_type == 0x9 and i + 1 < len(track_data):  # Note On
                        note = track_data[i]
                        velocity = track_data[i+1]
                        i += 2
                        
                        if note <= 127 and velocity <= 127:
                            time_sec = current_time * tempo / (ticks_per_beat * 1000000)
                            
                            if velocity > 0:
                                active_notes[note] = {'start': time_sec, 'velocity': velocity}
                            else:
                                # Note off via velocity 0
                                if note in active_notes:
                                    start_info = active_notes.pop(note)
                                    all_notes.append({
                                        'pitch': note,
                                        'note_name': midi_to_note_name(note),
                                        'note_class': midi_to_note_class(note),
                                        'start': start_info['start'],
                                        'end': time_sec,
                                        'duration': time_sec - start_info['start'],
                                        'velocity': start_info['velocity']
                                    })
                                    
                    elif event_type == 0x8 and i + 1 < len(track_data):  # Note Off
                        note = track_data[i]
                        i += 2
                        
                        if note <= 127 and note in active_notes:
                            time_sec = current_time * tempo / (ticks_per_beat * 1000000)
                            start_info = active_notes.pop(note)
                            all_notes.append({
                                'pitch': note,
                                'note_name': midi_to_note_name(note),
                                'note_class': midi_to_note_class(note),
                                'start': start_info['start'],
                                'end': time_sec,
                                'duration': time_sec - start_info['start'],
                                'velocity': start_info['velocity']
                            })
                            
                    elif event_type in [0xA, 0xB, 0xE] and i + 1 < len(track_data):
                        i += 2
                    elif event_type in [0xC, 0xD] and i < len(track_data):
                        i += 1
                        
                else:  # Running status
                    event_type = (running_status >> 4) & 0x0F
                    
                    if event_type == 0x9 and i < len(track_data):
                        note = status
                        velocity = track_data[i]
                        i += 1
                        
                        if note <= 127 and velocity <= 127:
                            time_sec = current_time * tempo / (ticks_per_beat * 1000000)
                            
                            if velocity > 0:
                                active_notes[note] = {'start': time_sec, 'velocity': velocity}
                            else:
                                if note in active_notes:
                                    start_info = active_notes.pop(note)
                                    all_notes.append({
                                        'pitch': note,
                                        'note_name': midi_to_note_name(note),
                                        'note_class': midi_to_note_class(note),
                                        'start': start_info['start'],
                                        'end': time_sec,
                                        'duration': time_sec - start_info['start'],
                                        'velocity': start_info['velocity']
                                    })
                                    
                    elif event_type == 0x8 and i < len(track_data):
                        note = status
                        i += 1
                        
                        if note <= 127 and note in active_notes:
                            time_sec = current_time * tempo / (ticks_per_beat * 1000000)
                            start_info = active_notes.pop(note)
                            all_notes.append({
                                'pitch': note,
                                'note_name': midi_to_note_name(note),
                                'note_class': midi_to_note_class(note),
                                'start': start_info['start'],
                                'end': time_sec,
                                'duration': time_sec - start_info['start'],
                                'velocity': start_info['velocity']
                            })
                            
                    elif event_type in [0xA, 0xB, 0xE] and i < len(track_data):
                        i += 1
                        
            except Exception as e:
                i += 1  # Skip problematic byte and continue
                continue
        
        pos = pos + 8 + track_length
    
    # Close any remaining active notes
    if all_notes:
        max_time = max(n['start'] for n in all_notes) + 1.0
        for note, info in active_notes.items():
            all_notes.append({
                'pitch': note,
                'note_name': midi_to_note_name(note),
                'note_class': midi_to_note_class(note),
                'start': info['start'],
                'end': max_time,
                'duration': max_time - info['start'],
                'velocity': info['velocity']
            })
    
    all_notes.sort(key=lambda x: (x['start'], x['pitch']))
    tempo_bpm = 60000000 / tempo
    
    return all_notes, tempo_bpm, ticks_per_beat


def analyze_midi_file(midi_path: str) -> Dict:
    """
    Analyze MIDI file - tries mido first, falls back to manual parsing
    """
    try:
        print("🎹 Analyzing MIDI file...")
        
        # Try manual parsing (more robust)
        try:
            notes, tempo_bpm, ticks_per_beat = parse_midi_manually(midi_path)
            print(f"   Manual parser: {len(notes)} notes, {tempo_bpm:.1f} BPM")
        except Exception as e:
            print(f"   Manual parser failed: {e}")
            # Try mido as fallback
            import mido
            mid = mido.MidiFile(midi_path)
            notes, tempo_bpm = extract_notes_with_mido(mid)
        
        if not notes:
            return {
                "total_notes": 0,
                "error": "No notes found in MIDI file"
            }
        
        print(f"   Found {len(notes)} notes")
        
        # Calculate duration
        duration = max(n.get('end', n['start'] + 0.5) for n in notes) if notes else 0
        print(f"   Duration: {duration:.2f} seconds")
        
        # Detect chords
        chords = detect_chords_improved(notes, onset_window=0.15, gap_threshold=0.4)
        print(f"   Found {len(chords)} chord groups")
        
        # Analyze each chord
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
        
        # Timing analysis
        timing_analysis = analyze_timing(notes)
        
        # Dynamics analysis
        dynamics_analysis = analyze_dynamics(notes)
        
        # Progression
        progression = detect_progression(chord_analysis, detected_key)
        
        result = {
            "total_notes": len(notes),
            "notes": notes[:100],
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


def extract_notes_with_mido(mid) -> Tuple[List[Dict], float]:
    """Extract notes using mido library"""
    import mido
    
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    
    seconds_per_tick = tempo / (ticks_per_beat * 1000000)
    tempo_bpm = mido.tempo2bpm(tempo)
    
    notes = []
    for track in mid.tracks:
        current_time = 0
        active_notes = {}
        
        for msg in track:
            current_time += msg.time
            time_sec = current_time * seconds_per_tick
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {'start': time_sec, 'velocity': msg.velocity}
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_info = active_notes.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'note_name': midi_to_note_name(msg.note),
                        'note_class': midi_to_note_class(msg.note),
                        'start': start_info['start'],
                        'end': time_sec,
                        'duration': time_sec - start_info['start'],
                        'velocity': start_info['velocity']
                    })
    
    notes.sort(key=lambda x: (x['start'], x['pitch']))
    return notes, tempo_bpm


def detect_chords_improved(notes: List[Dict], onset_window: float = 0.15, gap_threshold: float = 0.4) -> List[List[Dict]]:
    """
    Improved chord detection:
    - Groups notes that start within onset_window
    - Separates groups when there's a gap > gap_threshold
    """
    if not notes:
        return []
    
    # Group by onset time
    onset_groups = []
    current_group = [notes[0]]
    
    for note in notes[1:]:
        if abs(note['start'] - current_group[0]['start']) <= onset_window:
            current_group.append(note)
        else:
            if current_group:
                onset_groups.append(current_group)
            current_group = [note]
    
    if current_group:
        onset_groups.append(current_group)
    
    # Merge groups that are close, separate those with gaps
    if not onset_groups:
        return []
    
    chords = []
    current_chord = onset_groups[0]
    current_time = min(n['start'] for n in current_chord)
    
    for group in onset_groups[1:]:
        group_time = min(n['start'] for n in group)
        
        if group_time - current_time > gap_threshold:
            # New chord
            if len(current_chord) >= 2:
                chords.append(current_chord)
            current_chord = group
            current_time = group_time
        else:
            # Merge
            if len(current_chord) + len(group) <= 10:
                current_chord.extend(group)
            else:
                if len(current_chord) >= 2:
                    chords.append(current_chord)
                current_chord = group
                current_time = group_time
    
    if len(current_chord) >= 2:
        chords.append(current_chord)
    
    return chords


def analyze_chord(chord: List[Dict]) -> Dict:
    """Analyze a chord and identify its type"""
    sorted_notes = sorted(chord, key=lambda x: x['pitch'])
    pitches = [n['pitch'] for n in sorted_notes]
    
    pitch_classes = list(set(p % 12 for p in pitches))
    
    bass_note = pitches[0]
    bass_class = bass_note % 12
    bass_name = midi_to_note_class(bass_note, prefer_flat=True)
    
    chord_type, root, root_pc = identify_jazz_chord(pitch_classes, bass_class)
    
    note_classes = [midi_to_note_class(p, prefer_flat=True) for p in pitches]
    
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
        'end_time': max(n.get('end', n['start'] + 0.5) for n in chord),
        'num_notes': len(chord)
    }


def identify_jazz_chord(pitch_classes: List[int], bass_pc: int) -> Tuple[str, str, int]:
    """Identify jazz chord type"""
    pcs = set(pitch_classes)
    
    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    chord_templates = [
        ({0, 4, 7, 11}, "maj7"),
        ({0, 4, 7, 10}, "7"),
        ({0, 3, 7, 10}, "m7"),
        ({0, 3, 6, 10}, "m7b5"),
        ({0, 3, 6, 9}, "dim7"),
        ({0, 4, 7, 10, 2}, "9"),
        ({0, 4, 7, 10, 1}, "7b9"),
        ({0, 4, 7, 10, 3}, "7#9"),
        ({0, 4, 6, 10}, "7b5"),
        ({0, 4, 8, 10}, "7#5"),
        ({0, 3, 7, 10, 2}, "m9"),
        ({0, 3, 7, 11}, "mMaj7"),
        ({0, 4, 7, 11, 2}, "maj9"),
        ({0, 5, 7, 10}, "7sus4"),
        ({0, 4, 7}, ""),
        ({0, 3, 7}, "m"),
        ({0, 3, 6}, "dim"),
        ({0, 4, 8}, "aug"),
        ({0, 4, 7, 9}, "6"),
        ({0, 3, 7, 9}, "m6"),
        ({0, 4, 7, 10, 9}, "13"),
        ({0, 4, 7, 10, 2, 9}, "13"),
    ]
    
    best_match = ("?", None, bass_pc)
    best_score = 0
    
    for root_pc in range(12):
        intervals = set((pc - root_pc) % 12 for pc in pcs)
        
        for template, chord_type in chord_templates:
            matching = len(intervals & template)
            
            if template.issubset(intervals) or intervals.issubset(template):
                if matching > best_score:
                    best_score = matching
                    best_match = (chord_type, note_names[root_pc], root_pc)
            elif matching >= 3 and matching >= len(template) - 1:
                if matching > best_score:
                    best_score = matching
                    best_match = (chord_type, note_names[root_pc], root_pc)
    
    return best_match


def detect_key_from_notes(notes: List[Dict]) -> str:
    """Detect key from note distribution"""
    if not notes:
        return "C Major"
    
    note_classes = [n['note_class'] for n in notes]
    counter = Counter(note_classes)
    
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    note_order = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_key = "C Major"
    best_score = -1
    total = sum(counter.values())
    
    for root_idx, root in enumerate(note_order):
        dist = []
        for i in range(12):
            note = note_order[(root_idx + i) % 12]
            dist.append(counter.get(note, 0) / total if total > 0 else 0)
        
        try:
            major_corr = float(np.corrcoef(dist, major_profile)[0, 1])
            if major_corr > best_score:
                best_score = major_corr
                best_key = f"{root} Major"
            
            minor_corr = float(np.corrcoef(dist, minor_profile)[0, 1])
            if minor_corr > best_score:
                best_score = minor_corr
                best_key = f"{root} Minor"
        except:
            pass
    
    return best_key


def detect_progression(chords: List[Dict], key: str) -> Dict:
    """Detect chord progression"""
    if not chords:
        return {"progression": [], "analysis": "No chords detected", "type": "None"}
    
    symbols = [c['symbol'] for c in chords]
    types = [c.get('type', '') for c in chords]
    roots = [c.get('root', '') for c in chords]
    
    # Detect ii-V-I
    progression_type = "Custom"
    
    for i in range(len(types) - 2):
        if ('m7' in types[i] and 'm7b5' not in types[i] and
            types[i+1] in ['7', '7b9', '7#9', '7b5', '7#5', '9', '13'] and
            'maj7' in types[i+2]):
            progression_type = "ii-V-I"
            break
        if ('m7b5' in types[i] and
            types[i+1] in ['7', '7b9', '7#9', '7b5'] and
            'm7' in types[i+2] and 'maj' not in types[i+2]):
            progression_type = "ii-V-i (minor)"
            break
    
    # Roman numerals
    roman_numerals = []
    if "Major" in key or "Minor" in key:
        key_root = key.split()[0]
        is_major = "Major" in key
        for chord in chords:
            roman = get_roman_numeral(chord.get('root', ''), key_root, chord.get('type', ''), is_major)
            roman_numerals.append(roman)
    
    return {
        "symbols": symbols,
        "type": progression_type,
        "roman_numerals": roman_numerals,
        "analysis": f"{progression_type} in {key}" if progression_type != "Custom" else f"Progression in {key}"
    }


def get_roman_numeral(chord_root: str, key_root: str, chord_type: str, is_major: bool) -> str:
    """Convert chord to Roman numeral"""
    note_to_pc = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
        'A#': 10, 'Bb': 10, 'B': 11
    }
    
    key_pc = note_to_pc.get(key_root, 0)
    chord_pc = note_to_pc.get(chord_root, 0)
    degree = (chord_pc - key_pc) % 12
    
    numerals = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']
    numeral = numerals[degree]
    
    if chord_type and 'm' in chord_type and 'maj' not in chord_type.lower():
        numeral = numeral.lower()
    
    if chord_type and chord_type not in ['', 'm']:
        suffix = chord_type.replace('m', '') if chord_type.startswith('m') and 'maj' not in chord_type else chord_type
        numeral = f"{numeral}{suffix}"
    
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
    
    precision = max(0, 1.0 - (std_interval / mean_interval)) if mean_interval > 0 else 1.0
    
    return {
        "precision_score": min(1.0, precision),
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
    """Analyze voice leading between chords"""
    if len(chords) < 2:
        return []
    
    voice_leading = []
    for i in range(len(chords) - 1):
        c1, c2 = chords[i], chords[i + 1]
        common = set(c1.get('pitches', [])) & set(c2.get('pitches', []))
        
        voice_leading.append({
            'from': c1.get('symbol', '?'),
            'to': c2.get('symbol', '?'),
            'common_tones': len(common)
        })
    
    return voice_leading
