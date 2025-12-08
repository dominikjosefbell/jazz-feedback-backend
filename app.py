"""
Fixed note detection with Basic Pitch ENABLED
Replace the disabled section in app.py (lines 89-130) with this code
"""

import streamlit as st
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Global model variable
basic_pitch_model = None

def load_basic_pitch_model():
    """
    Load Basic Pitch model for note detection
    """
    try:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        
        logger.info("Loading Basic Pitch model...")
        model = Model(ICASSP_2022_MODEL_PATH)
        logger.info("✅ Basic Pitch model loaded successfully")
        return model
        
    except ImportError as e:
        logger.error(f"Basic Pitch not installed: {e}")
        raise Exception("Please install: pip install basic-pitch tensorflow")
    except Exception as e:
        logger.error(f"Error loading Basic Pitch model: {e}")
        raise

def detect_notes_basic_pitch(audio_file_path: str) -> dict:
    """
    Detect notes using Basic Pitch
    
    Returns:
        dict with:
        - notes: List of detected notes with (start_time, duration, pitch_midi, confidence)
        - note_count: Total number of notes
        - note_density: Notes per second
        - pitch_range: (min_pitch, max_pitch) in MIDI numbers
        - detected_chords: List of simultaneous notes (potential chords)
    """
    global basic_pitch_model
    
    # Load model if not already loaded
    if basic_pitch_model is None:
        basic_pitch_model = load_basic_pitch_model()
    
    try:
        from basic_pitch.inference import predict
        import librosa
        
        logger.info(f"Detecting notes in: {audio_file_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_file_path, sr=22050, mono=True)
        duration = len(audio) / sr
        
        # Run Basic Pitch prediction
        model_output, midi_data, note_events = predict(
            audio_path=audio_file_path,
            model_or_model_path=basic_pitch_model
        )
        
        # Extract note information
        notes = []
        for note in note_events:
            notes.append({
                'start_time': note['start_time_s'],
                'duration': note['duration_s'],
                'pitch_midi': note['pitch_midi'],
                'pitch_name': midi_to_note_name(note['pitch_midi']),
                'confidence': note['confidence'],
                'velocity': note.get('velocity', 127)
            })
        
        # Calculate statistics
        note_count = len(notes)
        note_density = note_count / duration if duration > 0 else 0
        
        pitch_values = [n['pitch_midi'] for n in notes]
        pitch_range = (min(pitch_values), max(pitch_values)) if pitch_values else (0, 0)
        
        # Detect potential chords (notes within 50ms of each other)
        chords = detect_simultaneous_notes(notes, time_window=0.05)
        
        logger.info(f"✅ Detected {note_count} notes ({note_density:.2f} notes/sec)")
        logger.info(f"✅ Pitch range: {midi_to_note_name(pitch_range[0])} - {midi_to_note_name(pitch_range[1])}")
        logger.info(f"✅ Detected {len(chords)} potential chords")
        
        return {
            'notes': notes,
            'note_count': note_count,
            'note_density': note_density,
            'pitch_range': pitch_range,
            'pitch_range_names': (midi_to_note_name(pitch_range[0]), midi_to_note_name(pitch_range[1])),
            'detected_chords': chords,
            'duration': duration,
            'midi_data': midi_data  # Full MIDI data for advanced analysis
        }
        
    except Exception as e:
        logger.error(f"Error in note detection: {e}")
        raise

def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI number to note name (e.g., 60 -> C4)"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def detect_simultaneous_notes(notes: list, time_window: float = 0.05) -> list:
    """
    Detect notes that happen at the same time (potential chords)
    
    Args:
        notes: List of note dictionaries
        time_window: Time window in seconds to consider notes simultaneous
        
    Returns:
        List of chords, each chord is a list of notes
    """
    if not notes:
        return []
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda x: x['start_time'])
    
    chords = []
    current_chord = [sorted_notes[0]]
    current_time = sorted_notes[0]['start_time']
    
    for note in sorted_notes[1:]:
        if note['start_time'] - current_time <= time_window:
            # Note is part of current chord
            current_chord.append(note)
        else:
            # Save current chord if it has multiple notes
            if len(current_chord) >= 2:
                chords.append(current_chord)
            # Start new chord
            current_chord = [note]
            current_time = note['start_time']
    
    # Don't forget the last chord
    if len(current_chord) >= 2:
        chords.append(current_chord)
    
    return chords

def analyze_chord(chord: list) -> dict:
    """
    Analyze a chord (list of simultaneous notes)
    
    Returns:
        dict with:
        - pitches: List of MIDI numbers
        - note_names: List of note names
        - root: Likely root note
        - intervals: Intervals from root
        - chord_type: Estimated chord type (maj7, min7, dom7, etc.)
    """
    pitches = sorted([n['pitch_midi'] for n in chord])
    note_names = [midi_to_note_name(p) for p in pitches]
    
    # Calculate intervals from lowest note (assumed root)
    root = pitches[0]
    intervals = [(p - root) % 12 for p in pitches]
    
    # Identify chord type based on intervals
    chord_type = identify_chord_type(intervals)
    
    return {
        'pitches': pitches,
        'note_names': note_names,
        'root': midi_to_note_name(root),
        'intervals': intervals,
        'chord_type': chord_type,
        'note_count': len(pitches)
    }

def identify_chord_type(intervals: list) -> str:
    """
    Identify chord type from intervals
    
    Examples:
    [0, 4, 7] = Major triad
    [0, 3, 7] = Minor triad
    [0, 4, 7, 11] = Major 7th
    [0, 3, 7, 10] = Minor 7th
    [0, 4, 7, 10] = Dominant 7th
    """
    intervals_set = set(intervals)
    
    # Common jazz chord types
    chord_types = {
        frozenset([0, 4, 7]): "Major",
        frozenset([0, 3, 7]): "Minor",
        frozenset([0, 3, 6]): "Diminished",
        frozenset([0, 4, 8]): "Augmented",
        frozenset([0, 4, 7, 11]): "Major 7",
        frozenset([0, 3, 7, 10]): "Minor 7",
        frozenset([0, 4, 7, 10]): "Dominant 7",
        frozenset([0, 3, 6, 10]): "Half-Diminished 7",
        frozenset([0, 3, 6, 9]): "Diminished 7",
        frozenset([0, 4, 7, 10, 14]): "Dominant 9",
        frozenset([0, 4, 7, 11, 14]): "Major 9",
        frozenset([0, 3, 7, 10, 14]): "Minor 9",
    }
    
    # Try to match
    for chord_intervals, name in chord_types.items():
        if chord_intervals.issubset(intervals_set):
            return name
    
    return "Unknown"

# Example usage in app.py:
def main():
    st.title("Jazz Piano Feedback App - WITH NOTE DETECTION")
    
    uploaded_file = st.file_uploader("Upload your piano recording", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_file:
        # Save file
        audio_path = f"/tmp/{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
        
        with st.spinner("🎹 Detecting notes..."):
            try:
                # DETECT NOTES!
                note_data = detect_notes_basic_pitch(audio_path)
                
                # Display results
                st.success(f"✅ Detected {note_data['note_count']} notes")
                st.write(f"**Note Density:** {note_data['note_density']:.2f} notes/second")
                st.write(f"**Pitch Range:** {note_data['pitch_range_names'][0]} to {note_data['pitch_range_names'][1]}")
                st.write(f"**Detected Chords:** {len(note_data['detected_chords'])}")
                
                # Analyze chords
                if note_data['detected_chords']:
                    st.subheader("🎵 Detected Chords:")
                    for i, chord in enumerate(note_data['detected_chords'][:5]):  # Show first 5
                        chord_info = analyze_chord(chord)
                        st.write(f"**Chord {i+1}:** {chord_info['root']} {chord_info['chord_type']}")
                        st.write(f"  Notes: {', '.join(chord_info['note_names'])}")
                
                # Now you can analyze if the notes are correct!
                # For example, if user said they're playing Dm7-G7-Cmaj7:
                # - Check if detected chords match
                # - Check voice leading
                # - Check guide tones
                # - Check timing
                
            except Exception as e:
                st.error(f"❌ Note detection failed: {e}")
                st.info("Make sure Basic Pitch is installed: pip install basic-pitch tensorflow")

if __name__ == "__main__":
    main()
