#!/usr/bin/env python3
"""
WE.ED.IT - Professionelle KI-Musikvideos
Automatisches Schneiden mit Art Director, Beat Sync und Vector Logic Matching

Workflow:
  1 x Song (MP3) + N x Clips (MP4) → 1 x Musikvideo (MP4)
  
Verzeichnisse:
  Audio: D:\Oidasheim\NFOs\mp3s
  Clips: D:\Oidasheim\NFOs\Clips
  Sync:  D:\Oidasheim\repos\cucl.dex
"""

import os
import json
import numpy as np
import librosa
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
from enum import Enum
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# KONFIGURATION
# ============================================================================

class Config:
    """Zentrale Konfiguration für WE.ED.IT"""
    
    # Verzeichnisse
    AUDIO_DIR = r"D:\Oidasheim\NFOs\mp3s"
    CLIPS_DIR = r"D:\Oidasheim\NFOs\Clips"
    SYNC_DIR = r"D:\Oidasheim\repos\cucl.dex"
    
    # Ausgabe
    OUTPUT_DIR = r"D:\Oidasheim\WE.ED.IT\output"
    CACHE_DIR = r"D:\Oidasheim\WE.ED.IT\cache"
    DB_DIR = r"D:\Oidasheim\WE.ED.IT\database"
    
    # Verarbeitung
    SAMPLE_RATE = 22050
    MIN_BEAT_CONFIDENCE = 0.6
    EMBEDDING_DIM = 384
    
    # AI Tags
    MOOD_TAGS = [
        'energetic', 'calm', 'aggressive', 'melancholic', 
        'joyful', 'dark', 'uplifting', 'introspective'
    ]
    
    VISUAL_TAGS = [
        'fast_cut', 'slow_motion', 'closeup', 'wide_shot',
        'handheld', 'static', 'drone', 'pov', 'silhouette',
        'high_contrast', 'neon', 'monochrome', 'colorful'
    ]


# ============================================================================
# AUDIO METADATA STRUCTURES
# ============================================================================

@dataclass
class AudioMetadata:
    """Umfassende Audio-Metadaten für MP3-Datei"""
    
    filename: str
    filepath: str
    duration_s: float
    bpm: float
    time_signature: str  # z.B. "4/4"
    
    # Struktur
    structure: List[Dict] = field(default_factory=list)  # [{label, start_s, end_s, energy}, ...]
    
    # Lyriken
    lyrics: Optional[str] = None
    lyric_themes: List[str] = field(default_factory=list)  # ['love', 'party', 'struggle', ...]
    
    # Semantik
    semantic_description: str = ""
    mood_tags: List[str] = field(default_factory=list)
    
    # Energie-Profile
    energy_curve: List[float] = field(default_factory=list)
    spectral_features: Dict = field(default_factory=dict)
    
    # Analyse-Metadaten
    analysis_date: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Konvertiere zu JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class VideoMetadata:
    """Umfassende Video-Metadaten für MP4-Clip"""
    
    filename: str
    filepath: str
    duration_s: float
    fps: float
    resolution: Tuple[int, int]
    
    # Visuelle Eigenschaften
    shot_scale: str  # 'extreme_closeup', 'closeup', 'medium', 'wide', 'extreme_wide'
    shot_composition: str  # 'rule_of_thirds', 'centered', 'low_angle', 'high_angle', ...
    camera_movement: str  # 'static', 'pan', 'tilt', 'dolly', 'handheld', 'tracking', 'drone'
    
    # Beleuchtung & Farbe
    lighting_type: str  # 'lowkey', 'highkey', 'backlighting', 'sidelighting', 'practical', 'neon'
    color_palette: List[Tuple[int, int, int]] = field(default_factory=list)  # Dominante RGB Farben
    
    # Bewegung & Aktion
    motion_intensity: float  # 0.0 - 1.0
    primary_action: str  # 'static', 'slow', 'moderate', 'fast', 'explosive'
    
    # Semantik
    emotional_tags: List[str] = field(default_factory=list)
    visual_tags: List[str] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)  # ['person', 'car', 'crowd', ...]
    
    # Qualität & Bewertung
    focus_score: float = 0.0  # 0.0 - 1.0 (Schärfe)
    composition_score: float = 0.0  # 0.0 - 1.0 (ästhetische Qualität)
    overall_quality: float = 0.0  # 0.0 - 1.0
    
    # Embedding für semantisches Matching
    embedding_vector: Optional[np.ndarray] = None
    
    # Analyse-Metadaten
    analysis_date: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_parameters: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Konvertiere zu Dictionary"""
        d = asdict(self)
        if self.embedding_vector is not None:
            d['embedding_vector'] = self.embedding_vector.tolist()
        return d
    
    def to_json(self) -> str:
        """Konvertiere zu JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ============================================================================
# AUDIO-ANALYSE ENGINE
# ============================================================================

class AudioAnalysisEngine:
    """KI-gesteuerte Audio-Analyse für Musikvideos"""
    
    def __init__(self):
        self.sr = Config.SAMPLE_RATE
        logger.info(f"AudioAnalysisEngine initialisiert (sr={self.sr})")
    
    def analyze_mp3(self, mp3_path: str) -> AudioMetadata:
        """Vollständige Audio-Analyse einer MP3-Datei"""
        
        logger.info(f"Analysiere: {mp3_path}")
        
        y, sr = librosa.load(mp3_path, sr=self.sr)
        duration_s = librosa.get_duration(y=y, sr=sr)
        
        # 1. BPM und Beat Grid
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_strength=onset_env)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        logger.info(f"  BPM: {bpm:.1f}")
        logger.info(f"  Duration: {duration_s:.1f}s")
        
        # 2. Song-Struktur (Intro, Verse, Chorus, Bridge, Outro)
        structure = self._detect_song_structure(y, sr, beat_times)
        
        # 3. Lyriken (falls vorhanden - ID3 Tag)
        lyrics, lyric_themes = self._extract_lyrics_and_themes(mp3_path)
        
        # 4. Energie-Profil
        energy_curve = self._compute_energy_curve(y, sr)
        
        # 5. Spektrale Features
        spectral_features = self._compute_spectral_features(y, sr)
        
        # 6. Semantische Beschreibung & Mood Tags
        semantic_desc, mood_tags = self._infer_semantic_info(
            energy_curve, spectral_features, bpm, lyric_themes
        )
        
        # Time Signature (vereinfacht)
        time_sig = "4/4"  # Standard für die meisten Pop/Hip-Hop Songs
        
        metadata = AudioMetadata(
            filename=Path(mp3_path).name,
            filepath=mp3_path,
            duration_s=duration_s,
            bpm=bpm,
            time_signature=time_sig,
            structure=structure,
            lyrics=lyrics,
            lyric_themes=lyric_themes,
            semantic_description=semantic_desc,
            mood_tags=mood_tags,
            energy_curve=energy_curve.tolist(),
            spectral_features=spectral_features
        )
        
        return metadata
    
    def _detect_song_structure(self, y: np.ndarray, sr: int, 
                               beat_times: np.ndarray) -> List[Dict]:
        """Erkennt Song-Struktur (Intro, Verse, Chorus, etc.)"""
        
        # Chroma-Feature für strukturelle Ähnlichkeit
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Segmentierung
        segments = librosa.sequence.viterbi(
            chroma,
            librosa.sequence.transition_uniform(3, chroma.shape[1])
        )
        
        segment_times = librosa.frames_to_time(segments, sr=sr)
        energy = librosa.feature.rms(y=y)[0]
        
        # Label basierend auf Energie und Position
        section_labels = []
        for i in range(len(segments) - 1):
            seg_energy = np.mean(energy[segments[i]:segments[i+1]])
            relative_energy = seg_energy / (np.max(energy) + 1e-8)
            
            # Vereinfachte Heuristik
            if len(section_labels) == 0 and relative_energy < 0.4:
                label = 'intro'
            elif relative_energy > 0.75:
                label = 'drop' if 'drop' not in section_labels else 'build'
            elif relative_energy > 0.5:
                label = 'chorus'
            elif relative_energy < 0.35:
                label = 'bridge' if 'bridge' in [s['label'] for s in section_labels] else 'verse'
            else:
                label = 'verse'
            
            section_labels.append(label)
        
        # Strukturiere als Liste
        structure = []
        for i in range(len(segments) - 1):
            start_s = segment_times[i]
            end_s = segment_times[i + 1]
            energy_val = np.mean(energy[segments[i]:segments[i+1]])
            
            structure.append({
                'label': section_labels[i],
                'start_s': float(start_s),
                'end_s': float(end_s),
                'energy': float(energy_val)
            })
        
        logger.info(f"  Struktur: {[s['label'] for s in structure]}")
        return structure
    
    def _extract_lyrics_and_themes(self, mp3_path: str) -> Tuple[Optional[str], List[str]]:
        """Extrahiere Lyriken und erkenne thematische Schlüsselwörter"""
        
        try:
            from mutagen.id3 import ID3
            
            tags = ID3(mp3_path)
            
            # Versuche unsynchronisierte Lyriken (USLT)
            lyrics = None
            sylt_frames = tags.getall('SYLT')
            
            if sylt_frames:
                # Nutze synchronisierte Lyriken
                lyric_texts = []
                for frame in sylt_frames:
                    for text, _ in frame.text:
                        lyric_texts.append(text)
                lyrics = '\n'.join(lyric_texts)
            else:
                # Fallback auf USLT
                uslt_frames = tags.getall('USLT')
                if uslt_frames:
                    lyrics = uslt_frames[0].text
            
            # Erkenne thematische Schlüsselwörter (vereinfachte Heuristik)
            themes = []
            if lyrics:
                lyrics_lower = lyrics.lower()
                
                # Einfache Keyword-Matching
                theme_keywords = {
                    'love': ['love', 'heart', 'together', 'forever', 'baby'],
                    'party': ['party', 'dance', 'night', 'club', 'celebrate'],
                    'struggle': ['fight', 'pain', 'hard', 'strong', 'never give up'],
                    'empowerment': ['power', 'rise', 'strong', 'own', 'believe'],
                    'introspection': ['lonely', 'alone', 'think', 'feel', 'inside'],
                    'celebration': ['yeah', 'yeah yeah', 'woah', 'uh', 'huh']
                }
                
                for theme, keywords in theme_keywords.items():
                    if any(kw in lyrics_lower for kw in keywords):
                        themes.append(theme)
            
            return lyrics, themes
        
        except Exception as e:
            logger.warning(f"Konnte Lyriken nicht extrahieren: {e}")
            return None, []
    
    def _compute_energy_curve(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Berechne Energie-Kurve über die Zeit"""
        
        S = np.abs(librosa.stft(y))
        energy = librosa.feature.rms(S=S)[0]
        
        # Normalisiere auf 0-1
        energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        
        return energy_normalized
    
    def _compute_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Berechne spektrale Merkmale"""
        
        S = np.abs(librosa.stft(y))
        
        # Spektrale Zentroide (Helligkeit)
        spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Spectral Flatness (Timbral Charakter)
        spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_flatness_std': float(np.std(spectral_flatness))
        }
    
    def _infer_semantic_info(self, energy_curve: np.ndarray, 
                            spectral_features: Dict, bpm: float,
                            lyric_themes: List[str]) -> Tuple[str, List[str]]:
        """Inferiere semantische Information und Mood Tags"""
        
        mood_tags = []
        
        # Energy-basiert
        mean_energy = np.mean(energy_curve)
        if mean_energy > 0.7:
            mood_tags.append('energetic')
        elif mean_energy < 0.3:
            mood_tags.append('calm')
        
        # BPM-basiert
        if bpm > 130:
            mood_tags.append('fast')
        elif bpm < 90:
            mood_tags.append('slow')
        
        # Spektral-basiert
        spec_centroid = spectral_features['spectral_centroid_mean']
        if spec_centroid > 3000:
            mood_tags.append('bright')
        elif spec_centroid < 2000:
            mood_tags.append('dark')
        
        # Lyrik-basiert
        for theme in lyric_themes:
            if theme in ['struggle', 'introspection']:
                mood_tags.extend(['introspective', 'deep'])
            elif theme in ['celebration', 'party']:
                mood_tags.extend(['uplifting', 'fun'])
        
        # Semantische Beschreibung
        desc_parts = [
            f"BPM: {bpm:.0f}",
            f"Energie: {mean_energy:.1%}",
            f"Themes: {', '.join(lyric_themes) if lyric_themes else 'instrumental'}"
        ]
        semantic_desc = " | ".join(desc_parts)
        
        return semantic_desc, list(set(mood_tags))  # Entferne Duplikate


# ============================================================================
# VIDEO-ANALYSE ENGINE
# ============================================================================

class VideoAnalysisEngine:
    """KI-gesteuerte Video-Analyse für Clip-Pool"""
    
    def __init__(self):
        logger.info("VideoAnalysisEngine initialisiert")
    
    def analyze_mp4(self, mp4_path: str, max_frames: int = 200) -> VideoMetadata:
        """Vollständige Video-Analyse einer MP4-Datei"""
        
        logger.info(f"Analysiere: {mp4_path}")
        
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0
        
        logger.info(f"  Duration: {duration_s:.1f}s, Resolution: {width}x{height}, FPS: {fps:.1f}")
        
        # Keyframes analysieren
        motion_intensities = []
        focus_scores = []
        color_palette = []
        
        sample_rate = max(1, total_frames // max_frames)
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if frame_num % sample_rate != 0:
                continue
            
            # Motion Intensity (Optical Flow)
            if prev_frame is not None:
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None,
                                                   0.5, 3, 15, 3, 5, 1.2, 0)
                motion_mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_intensities.append(np.mean(motion_mag) / 255.0)
            
            # Focus Score (Laplacian Variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            focus_scores.append(min(1.0, laplacian_var / 1000.0))
            
            # Dominant Color Palette
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            pixels = lab.reshape((-1, 3))
            pixels = np.float32(pixels)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            
            for center in centers:
                bgr = cv2.cvtColor(np.uint8([[[center[0], center[1], center[2]]]]), cv2.COLOR_LAB2BGR)[0][0]
                color_palette.append(tuple(map(int, bgr)))
            
            prev_frame = frame
        
        cap.release()
        
        # Durchschnittswerte
        motion_intensity = np.mean(motion_intensities) if motion_intensities else 0.0
        focus_score = np.mean(focus_scores) if focus_scores else 0.5
        
        # Klassifizierung basierend auf Bewegung und Fokus
        if motion_intensity > 0.6:
            primary_action = 'explosive'
        elif motion_intensity > 0.4:
            primary_action = 'fast'
        elif motion_intensity > 0.2:
            primary_action = 'moderate'
        else:
            primary_action = 'slow' if motion_intensity > 0.05 else 'static'
        
        # Shot Scale Inferenz (vereinfacht)
        aspect_ratio = width / height
        shot_scale = 'medium'  # Default
        
        # Composition und weitere Heuristiken
        shot_composition = 'centered'  # Default
        camera_movement = 'static'
        lighting_type = 'natural'
        
        # Visuelle Tags
        visual_tags = []
        if motion_intensity > 0.5:
            visual_tags.append('fast_cut')
        if motion_intensity < 0.1:
            visual_tags.append('static')
        if focus_score > 0.8:
            visual_tags.append('sharp')
        
        # Emotional Tags (vereinfacht)
        emotional_tags = []
        if motion_intensity > 0.6:
            emotional_tags.extend(['dynamic', 'energetic'])
        else:
            emotional_tags.extend(['calm', 'contemplative'])
        
        # Quality Score
        composition_score = (focus_score + motion_intensity) / 2
        overall_quality = (focus_score * 0.6 + (0.5 if motion_intensity > 0.1 else 0.0) * 0.4)
        
        metadata = VideoMetadata(
            filename=Path(mp4_path).name,
            filepath=mp4_path,
            duration_s=duration_s,
            fps=fps,
            resolution=(width, height),
            shot_scale=shot_scale,
            shot_composition=shot_composition,
            camera_movement=camera_movement,
            lighting_type=lighting_type,
            color_palette=list(set(color_palette))[:10],  # Top 10 Farben
            motion_intensity=motion_intensity,
            primary_action=primary_action,
            emotional_tags=emotional_tags,
            visual_tags=visual_tags,
            focus_score=focus_score,
            composition_score=composition_score,
            overall_quality=overall_quality,
            analysis_parameters={
                'max_frames': max_frames,
                'sample_rate': max(1, total_frames // max_frames),
                'total_frames_analyzed': len(motion_intensities)
            }
        )
        
        logger.info(f"  Quality: {overall_quality:.1%}, Motion: {motion_intensity:.1%}, Focus: {focus_score:.1%}")
        
        return metadata


# ============================================================================
# VECTOR LOGIC MATCHING ENGINE
# ============================================================================

class VectorLogicMatcher:
    """
    Vector Logic Matching Engine
    
    Beantwortet drei zentrale Fragen:
    1. "Hast du?" - Welche Clips passen zu einem Song-Segment?
    2. "Suchst du?" - Welche Song-Segmente passen zu einem Clip?
    3. "Brauchst du?" - Welche Clips und Song-Segmente fehlen noch?
    """
    
    def __init__(self):
        logger.info("VectorLogicMatcher initialisiert")
        self.audio_embeddings = {}
        self.video_embeddings = {}
    
    def compute_audio_embedding(self, audio_metadata: AudioMetadata) -> np.ndarray:
        """
        Konvertiere Audio-Metadaten in semantischen Vektor (384-dim)
        """
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Konstruiere Text-Repräsentation
            text_parts = [
                audio_metadata.semantic_description,
                ' '.join(audio_metadata.mood_tags),
                ' '.join(audio_metadata.lyric_themes),
                f"BPM {audio_metadata.bpm:.0f}",
            ]
            
            # Weitere Kontext aus Struktur
            for section in audio_metadata.structure:
                text_parts.append(f"{section['label']} section")
            
            combined_text = ' | '.join(text_parts)
            embedding = model.encode(combined_text, convert_to_tensor=True)
            
            return embedding.cpu().numpy()
        
        except ImportError:
            # Fallback: einfacher numerischer Vektor
            logger.warning("sentence-transformers nicht verfügbar, nutze Fallback")
            vector = np.zeros(Config.EMBEDDING_DIM)
            vector[0] = audio_metadata.bpm / 200.0
            return vector
    
    def compute_video_embedding(self, video_metadata: VideoMetadata) -> np.ndarray:
        """
        Konvertiere Video-Metadaten in semantischen Vektor (384-dim)
        """
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Konstruiere Text-Repräsentation
            text_parts = [
                video_metadata.shot_scale,
                video_metadata.camera_movement,
                video_metadata.lighting_type,
                video_metadata.primary_action,
                ' '.join(video_metadata.emotional_tags),
                ' '.join(video_metadata.visual_tags),
                ' '.join(video_metadata.objects_detected),
            ]
            
            combined_text = ' | '.join(text_parts)
            embedding = model.encode(combined_text, convert_to_tensor=True)
            
            return embedding.cpu().numpy()
        
        except ImportError:
            logger.warning("sentence-transformers nicht verfügbar, nutze Fallback")
            vector = np.zeros(Config.EMBEDDING_DIM)
            vector[0] = video_metadata.motion_intensity
            return vector
    
    def hast_du(self, song_metadata: AudioMetadata, 
                available_clips: List[VideoMetadata]) -> List[Tuple[VideoMetadata, float]]:
        """
        "Hast du?" - Finde Clips, die zu einem Song-Segment passen
        
        Returns: Liste von (clip, match_score) sortiert nach Score
        """
        
        song_embedding = self.compute_audio_embedding(song_metadata)
        
        matches = []
        for clip in available_clips:
            clip_embedding = self.compute_video_embedding(clip)
            
            # Cosine Similarity
            similarity = np.dot(song_embedding, clip_embedding) / (
                np.linalg.norm(song_embedding) * np.linalg.norm(clip_embedding) + 1e-8
            )
            
            # Berücksichtige auch Energie-Matching
            if song_metadata.structure:
                avg_energy = np.mean([s['energy'] for s in song_metadata.structure])
                energy_match = 1.0 - abs(avg_energy - clip.motion_intensity)
                similarity = 0.7 * similarity + 0.3 * energy_match
            
            matches.append((clip, float(similarity)))
        
        # Sortiere nach Score (absteigend)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def suchst_du(self, clip_metadata: VideoMetadata,
                  available_songs: List[AudioMetadata]) -> List[Tuple[AudioMetadata, float]]:
        """
        "Suchst du?" - Finde Song-Segmente, die zu einem Clip passen
        
        Returns: Liste von (song, match_score) sortiert nach Score
        """
        
        clip_embedding = self.compute_video_embedding(clip_metadata)
        
        matches = []
        for song in available_songs:
            song_embedding = self.compute_audio_embedding(song_metadata)
            
            # Cosine Similarity
            similarity = np.dot(clip_embedding, song_embedding) / (
                np.linalg.norm(clip_embedding) * np.linalg.norm(song_embedding) + 1e-8
            )
            
            matches.append((song, float(similarity)))
        
        # Sortiere nach Score (absteigend)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def brauchst_du(self, song_metadata: AudioMetadata,
                    available_clips: List[VideoMetadata]) -> Dict:
        """
        "Brauchst du?" - Analysiere, welche Clips und Momente noch fehlen
        
        Returns: Analyse der fehlenden Inhalte
        """
        
        analysis = {
            'song_structure': song_metadata.structure,
            'coverage': {},
            'missing_elements': [],
            'recommendations': []
        }
        
        # Für jedes Song-Segment: finde beste Clips
        for i, segment in enumerate(song_metadata.structure):
            # Erstelle temporäre Song-Metadaten für Segment
            segment_audio = AudioMetadata(
                filename=f"{song_metadata.filename}_seg_{i}",
                filepath=song_metadata.filepath,
                duration_s=segment['end_s'] - segment['start_s'],
                bpm=song_metadata.bpm,
                time_signature=song_metadata.time_signature,
                structure=[segment],
                mood_tags=[],
                semantic_description=f"{segment['label']} section"
            )
            
            # Finde passende Clips
            matches = self.hast_du(segment_audio, available_clips)
            
            analysis['coverage'][segment['label']] = {
                'best_match': matches[0] if matches else None,
                'top_3_matches': matches[:3]
            }
            
            # Identifiziere fehlende Inhalte
            if not matches or matches[0][1] < 0.5:
                analysis['missing_elements'].append({
                    'segment': segment['label'],
                    'required_action': 'fast' if segment['energy'] > 0.6 else 'moderate'
                })
        
        return analysis


# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================

class WEEDITOrchestrator:
    """Zentrale Orchestrierungs-Engine für WE.ED.IT"""
    
    def __init__(self):
        self.audio_engine = AudioAnalysisEngine()
        self.video_engine = VideoAnalysisEngine()
        self.matcher = VectorLogicMatcher()
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.DB_DIR, exist_ok=True)
        
        logger.info("WE.ED.IT Orchestrator initialisiert")
    
    def process_audio_library(self, audio_dir: str = None) -> Dict[str, AudioMetadata]:
        """Verarbeite alle MP3-Dateien in einem Verzeichnis"""
        
        if audio_dir is None:
            audio_dir = Config.AUDIO_DIR
        
        logger.info(f"Verarbeite Audio-Bibliothek: {audio_dir}")
        
        audio_metadata_dict = {}
        
        if not os.path.exists(audio_dir):
            logger.warning(f"Audio-Verzeichnis nicht gefunden: {audio_dir}")
            return audio_metadata_dict
        
        mp3_files = list(Path(audio_dir).glob("*.mp3"))
        logger.info(f"Gefunden: {len(mp3_files)} MP3-Dateien")
        
        for mp3_path in mp3_files:
            try:
                metadata = self.audio_engine.analyze_mp3(str(mp3_path))
                audio_metadata_dict[mp3_path.name] = metadata
                
                # Speichere Metadaten
                cache_path = os.path.join(Config.CACHE_DIR, f"{mp3_path.stem}_audio.json")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(metadata.to_json())
                
                logger.info(f"  ✓ {mp3_path.name} ({metadata.bpm:.1f} BPM)")
            
            except Exception as e:
                logger.error(f"  ✗ Fehler bei {mp3_path.name}: {e}")
        
        return audio_metadata_dict
    
    def process_video_library(self, video_dir: str = None) -> Dict[str, VideoMetadata]:
        """Verarbeite alle MP4-Dateien in einem Verzeichnis"""
        
        if video_dir is None:
            video_dir = Config.CLIPS_DIR
        
        logger.info(f"Verarbeite Video-Bibliothek: {video_dir}")
        
        video_metadata_dict = {}
        
        if not os.path.exists(video_dir):
            logger.warning(f"Video-Verzeichnis nicht gefunden: {video_dir}")
            return video_metadata_dict
        
        mp4_files = list(Path(video_dir).glob("**/*.mp4"))
        logger.info(f"Gefunden: {len(mp4_files)} MP4-Dateien")
        
        for mp4_path in mp4_files[:10]:  # Begrenzen zur Demo
            try:
                metadata = self.video_engine.analyze_mp4(str(mp4_path))
                video_metadata_dict[mp4_path.name] = metadata
                
                # Speichere Metadaten
                cache_path = os.path.join(Config.CACHE_DIR, f"{mp4_path.stem}_video.json")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(metadata.to_json())
                
                logger.info(f"  ✓ {mp4_path.name} (Qualität: {metadata.overall_quality:.1%})")
            
            except Exception as e:
                logger.error(f"  ✗ Fehler bei {mp4_path.name}: {e}")
        
        return video_metadata_dict
    
    def generate_matching_report(self, audio_metadata: AudioMetadata,
                                available_clips: List[VideoMetadata]) -> str:
        """Generiere einen Vector Logic Matching Report"""
        
        report = f"""
================================================================================
WE.ED.IT - VECTOR LOGIC MATCHING REPORT
================================================================================

SONG: {audio_metadata.filename}
BPM: {audio_metadata.bpm:.1f}
DURATION: {audio_metadata.duration_s:.1f}s
STRUCTURE: {', '.join([s['label'] for s in audio_metadata.structure])}
MOOD: {', '.join(audio_metadata.mood_tags)}
THEMES: {', '.join(audio_metadata.lyric_themes) if audio_metadata.lyric_themes else 'Instrumental'}

--------------------------------------------------------------------------------
1. HAST DU? - Beste Clip-Matches für diesen Song
--------------------------------------------------------------------------------

"""
        
        matches = self.matcher.hast_du(audio_metadata, available_clips)
        
        for i, (clip, score) in enumerate(matches[:5], 1):
            report += f"\n{i}. {clip.filename} (Score: {score:.1%})\n"
            report += f"   Action: {clip.primary_action} | Motion: {clip.motion_intensity:.1%} | Focus: {clip.focus_score:.1%}\n"
            report += f"   Tags: {', '.join(clip.emotional_tags + clip.visual_tags)}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report


def main():
    """Haupteinstiegspunkt für WE.ED.IT"""
    
    logger.info("="*80)
    logger.info("WE.ED.IT - Automatisches KI-Musikvideo-System")
    logger.info("="*80)
    
    orchestrator = WEEDITOrchestrator()
    
    # Verarbeite Audio- und Video-Bibliotheken
    logger.info("\n📻 Verarbeite Audio-Bibliothek...")
    audio_dict = orchestrator.process_audio_library()
    
    logger.info("\n🎬 Verarbeite Video-Bibliothek...")
    video_dict = orchestrator.process_video_library()
    
    # Demonstriere Vector Logic Matching
    if audio_dict and video_dict:
        first_song = list(audio_dict.values())[0]
        available_clips = list(video_dict.values())
        
        logger.info("\n🎯 Generiere Matching Report...")
        report = orchestrator.generate_matching_report(first_song, available_clips)
        print(report)
        
        # Speichere Report
        report_path = os.path.join(Config.OUTPUT_DIR, "matching_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report gespeichert: {report_path}")
    
    logger.info("\n✅ Verarbeitung abgeschlossen!")


if __name__ == "__main__":
    main()
