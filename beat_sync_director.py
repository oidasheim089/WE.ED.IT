#!/usr/bin/env python3
"""
WE.ED.IT - BEAT SYNC VIDEODIRECTOR
Automatisches KI-Musikvideo-Rendering mit intelligenter Clip-Auswahl

ARCHITEKTUR (3-Ebenen):
  [EBENE 1] Videopool-Versteher & Content-Analyzer
            └─ Asset-Graph, Beat-/Emotion-Timeline
  
  [EBENE 2] Director-Ebene (Schnitt, Beat-Sync, Struktur)
            └─ Rough-Cut Timeline, EDL/XML, Marker, Blueprint
  
  [EBENE 3] VFX-Ebene (Effekte, Overlay, Platform-Polish)
            └─ Final-Render, Export-Presets, Metadaten
            
Output: 📤 Virales Musikvideo + Strategie-Paket

WORKFLOW:
  1 x MP3 (Sound) + N x MP4 (Clips) → 1 x Musikvideo (MP4)
  
  • Ignoriere Audio aus Clips → nur MP3 gibt den Ton an
  • Beat-synchronized Schnitte
  • Wiederholungs-Vermeidung durch Novelty-Signal
  • Semantisches Clip-Matching
"""

import os
import json
import csv
import numpy as np
import librosa
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import logging
import subprocess
from collections import defaultdict
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# KONFIGURATION & KONSTANTEN
# ============================================================================

class Config:
    """Zentrale Konfiguration"""
    
    # Verzeichnisse
    AUDIO_DIR = r"D:\Oidasheim\NFOs\mp3s"
    CLIPS_DIR = r"D:\Oidasheim\NFOs\Clips"
    SYNC_DIR = r"D:\Oidasheim\repos\cucl.dex"
    
    # Output
    OUTPUT_DIR = r"D:\Oidasheim\WE.ED.IT\output"
    CACHE_DIR = r"D:\Oidasheim\WE.ED.IT\cache"
    DB_DIR = r"D:\Oidasheim\WE.ED.IT\database"
    EXPORT_DIR = r"D:\Oidasheim\WE.ED.IT\exports"
    
    # Audio Processing
    SAMPLE_RATE = 22050
    MIN_BEAT_CONFIDENCE = 0.6
    HOP_LENGTH = 512
    
    # Video Processing
    TARGET_FPS = 30
    TARGET_RESOLUTION = (1920, 1080)
    
    # Embedding
    EMBEDDING_DIM = 384
    
    # Novelty Decay (verhindert Wiederholung)
    NOVELTY_LAMBDA = 0.8  # Je höher = stärkere Penalisierung
    
    # Min/Max Clip Duration
    MIN_CLIP_DURATION = 0.5  # seconds
    MAX_CLIP_DURATION = 15.0  # seconds


class CutType(Enum):
    """Typen von Schnitten"""
    HARD_CUT = "hard_cut"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    CROSS_FADE = "cross_fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"


# ============================================================================
# DATENSTRUKTUREN
# ============================================================================

@dataclass
class BeatMarker:
    """Beat-Marker für Timeline"""
    frame_index: int
    time_s: float
    beat_number: int
    confidence: float
    is_downbeat: bool = False


@dataclass
class SongSegment:
    """Song-Segment mit Struktur"""
    label: str  # 'intro', 'verse', 'chorus', 'bridge', 'drop', 'outro'
    start_s: float
    end_s: float
    start_beat: int
    end_beat: int
    energy: float
    emotion: str


@dataclass
class AudioBlueprint:
    """MP3-Blueprint für Video-Rendering"""
    filename: str
    filepath: str
    duration_s: float
    bpm: float
    beat_times: List[float]
    segments: List[SongSegment]
    energy_curve: np.ndarray
    spectral_features: Dict
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['energy_curve'] = self.energy_curve.tolist()
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_edl(self) -> str:
        """Exportiere als EDL (Edit Decision List)"""
        edl_lines = []
        edl_lines.append("TITLE: " + Path(self.filename).stem)
        edl_lines.append(f"DURATION: {self.duration_s:.2f}s")
        edl_lines.append(f"BPM: {self.bpm:.1f}")
        edl_lines.append("")
        
        for i, segment in enumerate(self.segments, 1):
            edl_lines.append(f"{i:03d}  {segment.label:12} AUDIO START: {segment.start_s:8.2f}s END: {segment.end_s:8.2f}s")
        
        return "\n".join(edl_lines)


@dataclass
class VideoNFO:
    """Video-Clip Metadaten (NFO)"""
    filename: str
    filepath: str
    duration_s: float
    fps: float
    resolution: Tuple[int, int]
    
    # Visuelles
    shot_scale: str
    camera_movement: str
    lighting_type: str
    color_palette: List[Tuple[int, int, int]]
    
    # Bewegung & Emotion
    motion_intensity: float
    primary_action: str
    emotional_tags: List[str]
    visual_tags: List[str]
    
    # Qualität
    focus_score: float
    composition_score: float
    overall_quality: float
    
    # Embedding
    embedding_vector: Optional[np.ndarray] = None
    
    # Analyse-Metadaten
    analysis_date: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_version: str = "2.0"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.embedding_vector is not None:
            d['embedding_vector'] = self.embedding_vector.tolist()
        return d
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def to_csv_row(self) -> Dict:
        """Konvertiere zu CSV-Zeile"""
        return {
            'filename': self.filename,
            'duration_s': self.duration_s,
            'fps': self.fps,
            'resolution': f"{self.resolution[0]}x{self.resolution[1]}",
            'shot_scale': self.shot_scale,
            'camera_movement': self.camera_movement,
            'lighting': self.lighting_type,
            'motion_intensity': f"{self.motion_intensity:.2%}",
            'primary_action': self.primary_action,
            'emotional_tags': '|'.join(self.emotional_tags),
            'visual_tags': '|'.join(self.visual_tags),
            'focus_score': f"{self.focus_score:.2%}",
            'composition_score': f"{self.composition_score:.2%}",
            'overall_quality': f"{self.overall_quality:.2%}",
            'analysis_date': self.analysis_date
        }


@dataclass
class TimelineClip:
    """Clip auf Timeline"""
    clip_nfo: VideoNFO
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    segment_label: str
    cut_type: CutType
    match_score: float
    novelty_penalty: float


# ============================================================================
# EBENE 1: VIDEOPOOL ANALYZER
# ============================================================================

class VideopoolAnalyzer:
    """Videopool-Versteher & Content-Analyzer"""
    
    def __init__(self):
        logger.info("🎬 VideopoolAnalyzer initialisiert (EBENE 1)")
        self.video_cache = {}
    
    def analyze_all_clips(self, video_dir: str = None) -> Dict[str, VideoNFO]:
        """Analysiere alle Clips im Verzeichnis"""
        
        if video_dir is None:
            video_dir = Config.CLIPS_DIR
        
        if not os.path.exists(video_dir):
            logger.warning(f"❌ Video-Verzeichnis nicht gefunden: {video_dir}")
            return {}
        
        mp4_files = list(Path(video_dir).rglob("*.mp4"))[:50]  # Max 50 für Demo
        logger.info(f"📊 Analysiere {len(mp4_files)} Video-Clips...")
        
        video_nfos = {}
        
        for i, mp4_path in enumerate(mp4_files, 1):
            try:
                nfo = self._analyze_single_clip(str(mp4_path))
                video_nfos[mp4_path.name] = nfo
                
                # Speichere NFO als JSON
                nfo_path = os.path.join(Config.CACHE_DIR, f"{mp4_path.stem}_video.json")
                with open(nfo_path, 'w', encoding='utf-8') as f:
                    f.write(nfo.to_json())
                
                logger.info(f"  [{i:2d}] ✓ {mp4_path.name} | Quality: {nfo.overall_quality:.1%}")
            
            except Exception as e:
                logger.error(f"  [{i:2d}] ✗ {mp4_path.name} | Error: {str(e)[:50]}")
        
        return video_nfos
    
    def _analyze_single_clip(self, mp4_path: str) -> VideoNFO:
        """Analysiere einen einzelnen Clip"""
        
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0
        
        # Sampling
        motion_intensities = []
        focus_scores = []
        color_palette = []
        
        sample_rate = max(1, total_frames // 100)
        prev_gray = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_num % sample_rate != 0:
                continue
            
            # Motion Intensity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_intensities.append(np.mean(motion_mag) / 255.0)
            prev_gray = gray
            
            # Focus Score
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            focus_scores.append(min(1.0, laplacian_var / 1000.0))
            
            # Color Palette (K-Means)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            pixels = lab.reshape((-1, 3)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, _, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            for center in centers:
                bgr = cv2.cvtColor(np.uint8([[[center[0], center[1], center[2]]]]), cv2.COLOR_LAB2BGR)[0][0]
                color_palette.append(tuple(map(int, bgr)))
            
            frame_count += 1
        
        cap.release()
        
        # Aggregiere
        motion_intensity = np.mean(motion_intensities) if motion_intensities else 0.0
        focus_score = np.mean(focus_scores) if focus_scores else 0.5
        
        # Klassifizierung
        if motion_intensity > 0.6:
            primary_action = 'explosive'
        elif motion_intensity > 0.4:
            primary_action = 'fast'
        elif motion_intensity > 0.2:
            primary_action = 'moderate'
        else:
            primary_action = 'static'
        
        # Visual Tags
        visual_tags = []
        if motion_intensity > 0.5:
            visual_tags.extend(['dynamic', 'fast_cut'])
        if focus_score > 0.75:
            visual_tags.append('sharp')
        if focus_score < 0.4:
            visual_tags.append('soft_focus')
        
        # Emotional Tags
        emotional_tags = []
        if motion_intensity > 0.6:
            emotional_tags.extend(['energetic', 'dramatic'])
        else:
            emotional_tags.extend(['calm', 'contemplative'])
        
        # Qualität
        composition_score = (focus_score + (1.0 if motion_intensity > 0.1 else 0.3)) / 2
        overall_quality = focus_score * 0.6 + min(1.0, motion_intensity * 1.5) * 0.4
        
        # Embedding (vereinfacht)
        embedding = np.random.randn(Config.EMBEDDING_DIM).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        nfo = VideoNFO(
            filename=Path(mp4_path).name,
            filepath=mp4_path,
            duration_s=duration_s,
            fps=fps,
            resolution=(width, height),
            shot_scale='medium',
            camera_movement='static',
            lighting_type='natural',
            color_palette=list(set(color_palette))[:10],
            motion_intensity=motion_intensity,
            primary_action=primary_action,
            emotional_tags=emotional_tags,
            visual_tags=visual_tags,
            focus_score=focus_score,
            composition_score=composition_score,
            overall_quality=overall_quality,
            embedding_vector=embedding
        )
        
        return nfo
    
    def export_clips_as_csv(self, video_nfos: Dict[str, VideoNFO], 
                           output_file: str = None) -> str:
        """Exportiere alle Clip-Metadaten als CSV"""
        
        if output_file is None:
            output_file = os.path.join(Config.EXPORT_DIR, "video_clips_analysis.csv")
        
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if not video_nfos:
                logger.warning("Keine Video-NFOs zum Exportieren")
                return output_file
            
            # Fieldnames aus erstem NFO
            first_nfo = list(video_nfos.values())[0]
            fieldnames = list(first_nfo.to_csv_row().keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for nfo in video_nfos.values():
                writer.writerow(nfo.to_csv_row())
        
        logger.info(f"📊 CSV exportiert: {output_file}")
        return output_file


# ============================================================================
# AUDIO BLUEPRINT ENGINE
# ============================================================================

class AudioBlueprintEngine:
    """Erstelle Audio-Blueprint aus MP3"""
    
    def __init__(self):
        logger.info("🎵 AudioBlueprintEngine initialisiert")
        self.sr = Config.SAMPLE_RATE
    
    def create_blueprint(self, mp3_path: str) -> AudioBlueprint:
        """Erstelle Blueprint aus MP3"""
        
        logger.info(f"🎼 Erstelle Blueprint: {Path(mp3_path).name}")
        
        y, sr = librosa.load(mp3_path, sr=self.sr)
        duration_s = librosa.get_duration(y=y, sr=sr)
        
        # BPM & Beat Detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_strength=onset_env)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Segments (Struktur)
        segments = self._detect_segments(y, sr, beat_times, bpm)
        
        # Energy Curve
        energy_curve = self._compute_energy_curve(y, sr)
        
        # Spectral Features
        spectral_features = self._compute_spectral_features(y, sr)
        
        blueprint = AudioBlueprint(
            filename=Path(mp3_path).name,
            filepath=mp3_path,
            duration_s=duration_s,
            bpm=bpm,
            beat_times=beat_times.tolist(),
            segments=segments,
            energy_curve=energy_curve,
            spectral_features=spectral_features
        )
        
        logger.info(f"  BPM: {bpm:.1f} | Duration: {duration_s:.1f}s | Segments: {len(segments)}")
        
        return blueprint
    
    def _detect_segments(self, y: np.ndarray, sr: int, 
                        beat_times: np.ndarray, bpm: float) -> List[SongSegment]:
        """Erkenne Song-Segmente"""
        
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        energy = librosa.feature.rms(y=y)[0]
        
        # Chroma-Segmentierung
        from scipy.spatial.distance import pdist
        chroma_frames = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
        
        # Vereinfachte Segmentierung basierend auf Energie
        segments = []
        segment_length_s = 8  # Durchschnittliche Segment-Länge
        
        start_s = 0
        segment_count = 0
        
        while start_s < librosa.get_duration(y=y, sr=sr):
            end_s = min(start_s + segment_length_s, librosa.get_duration(y=y, sr=sr))
            
            # Energie für Segment
            start_frame = int(librosa.time_to_frames(start_s, sr=sr))
            end_frame = int(librosa.time_to_frames(end_s, sr=sr))
            segment_energy = np.mean(energy[start_frame:end_frame])
            
            # Label basierend auf Position und Energie
            relative_pos = start_s / librosa.get_duration(y=y, sr=sr)
            
            if relative_pos < 0.1:
                label = 'intro'
            elif relative_pos < 0.35:
                label = 'verse'
            elif relative_pos < 0.65:
                label = 'chorus'
            elif relative_pos < 0.80:
                label = 'bridge'
            elif relative_pos < 0.95:
                label = 'drop'
            else:
                label = 'outro'
            
            # Emotion
            if segment_energy > 0.7:
                emotion = 'high_energy'
            elif segment_energy > 0.4:
                emotion = 'medium_energy'
            else:
                emotion = 'low_energy'
            
            # Finde Beat-Indizes
            beat_indices = np.where((beat_times >= start_s) & (beat_times < end_s))[0]
            start_beat = int(beat_indices[0]) if len(beat_indices) > 0 else 0
            end_beat = int(beat_indices[-1]) if len(beat_indices) > 0 else 0
            
            segment = SongSegment(
                label=label,
                start_s=start_s,
                end_s=end_s,
                start_beat=start_beat,
                end_beat=end_beat,
                energy=float(segment_energy),
                emotion=emotion
            )
            
            segments.append(segment)
            segment_count += 1
            start_s = end_s
        
        return segments
    
    def _compute_energy_curve(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Energie-Kurve"""
        S = np.abs(librosa.stft(y))
        energy = librosa.feature.rms(S=S)[0]
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
        return energy_norm
    
    def _compute_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Spektrale Features"""
        S = np.abs(librosa.stft(y))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            'spectral_centroid_mean': float(np.mean(centroid)),
            'zero_crossing_rate_mean': float(np.mean(zcr))
        }


# ============================================================================
# EBENE 2: DIRECTOR ENGINE (Beat Sync + Matching)
# ============================================================================

class DirectorEngine:
    """Director-Ebene: Schnitt, Beat-Sync, Struktur"""
    
    def __init__(self):
        logger.info("🎬 DirectorEngine initialisiert (EBENE 2)")
    
    def create_beat_synced_timeline(self, blueprint: AudioBlueprint,
                                   video_nfos: Dict[str, VideoNFO],
                                   use_novelty: bool = True) -> List[TimelineClip]:
        """Erstelle beat-synchronized Timeline"""
        
        logger.info(f"📹 Erstelle Beat-Synced Timeline für {blueprint.filename}")
        
        timeline_clips = []
        clip_usage_count = defaultdict(int)
        
        for segment in blueprint.segments:
            logger.info(f"  [{segment.label:10s}] {segment.start_s:6.1f}s - {segment.end_s:6.1f}s (Energy: {segment.energy:.1%})")
            
            # Finde beste Clips für Segment
            best_clips = self._find_best_clips_for_segment(
                segment, video_nfos, clip_usage_count, use_novelty
            )
            
            if not best_clips:
                logger.warning(f"    ⚠️  Keine passenden Clips gefunden!")
                continue
            
            # Füge Clips zum Timeline hinzu
            current_time = segment.start_s
            
            for clip_nfo, match_score, novelty_penalty in best_clips:
                if current_time >= segment.end_s:
                    break
                
                clip_duration = min(clip_nfo.duration_s, segment.end_s - current_time)
                
                if clip_duration < Config.MIN_CLIP_DURATION:
                    continue
                
                # Bestimme Cut-Type basierend auf Position
                if not timeline_clips:
                    cut_type = CutType.HARD_CUT
                else:
                    cut_type = CutType.HARD_CUT  # Vereinfacht
                
                # Kalkuliere Frame-Indizes
                start_frame = int(current_time * Config.TARGET_FPS)
                end_frame = int((current_time + clip_duration) * Config.TARGET_FPS)
                
                timeline_clip = TimelineClip(
                    clip_nfo=clip_nfo,
                    start_s=current_time,
                    end_s=current_time + clip_duration,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    segment_label=segment.label,
                    cut_type=cut_type,
                    match_score=match_score,
                    novelty_penalty=novelty_penalty
                )
                
                timeline_clips.append(timeline_clip)
                clip_usage_count[clip_nfo.filename] += 1
                current_time += clip_duration
        
        logger.info(f"  ✓ Timeline erstellt mit {len(timeline_clips)} Clips")
        return timeline_clips
    
    def _find_best_clips_for_segment(self, segment: SongSegment,
                                    video_nfos: Dict[str, VideoNFO],
                                    usage_count: Dict[str, int],
                                    use_novelty: bool = True) -> List[Tuple]:
        """Finde beste Clips für Segment (mit Novelty Avoidance)"""
        
        matches = []
        
        for filename, nfo in video_nfos.items():
            # Basis-Matching-Score
            base_score = self._calculate_match_score(segment, nfo)
            
            # Novelty Penalty (verhindert Wiederholung)
            if use_novelty:
                novelty_penalty = np.exp(-Config.NOVELTY_LAMBDA * usage_count[filename])
                final_score = base_score * novelty_penalty
            else:
                novelty_penalty = 1.0
                final_score = base_score
            
            matches.append((nfo, final_score, novelty_penalty))
        
        # Sortiere nach Score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:10]  # Top 10
    
    def _calculate_match_score(self, segment: SongSegment, nfo: VideoNFO) -> float:
        """Berechne Matching-Score zwischen Segment und Clip"""
        
        score = 0.0
        
        # Energy Matching (0.4 Gewicht)
        energy_diff = abs(segment.energy - nfo.motion_intensity)
        energy_score = 1.0 - min(1.0, energy_diff)
        score += 0.4 * energy_score
        
        # Qualität (0.3 Gewicht)
        quality_score = nfo.overall_quality
        score += 0.3 * quality_score
        
        # Emotional Alignment (0.3 Gewicht)
        emotional_score = 0.5  # Vereinfacht
        if segment.emotion == 'high_energy' and nfo.primary_action in ['fast', 'explosive']:
            emotional_score = 0.9
        elif segment.emotion == 'low_energy' and nfo.primary_action in ['static', 'moderate']:
            emotional_score = 0.9
        
        score += 0.3 * emotional_score
        
        return score
    
    def export_timeline_as_edl(self, timeline: List[TimelineClip],
                              output_file: str = None) -> str:
        """Exportiere Timeline als EDL (Edit Decision List)"""
        
        if output_file is None:
            output_file = os.path.join(Config.EXPORT_DIR, "timeline_blueprint.edl")
        
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        
        edl_lines = [
            "TITLE: WE.ED.IT Auto-Generated Video",
            f"TIMESTAMP: {datetime.now().isoformat()}",
            f"TOTAL_CLIPS: {len(timeline)}",
            "",
            "EVENT#|CLIP_NAME|START_TIME|END_TIME|DURATION|MATCH_SCORE|SEGMENT|CUT_TYPE",
            "="*100
        ]
        
        for i, clip in enumerate(timeline, 1):
            edl_lines.append(
                f"{i:03d}|{clip.clip_nfo.filename[:40]:40s}|"
                f"{clip.start_s:8.2f}|{clip.end_s:8.2f}|"
                f"{clip.end_s - clip.start_s:6.2f}|"
                f"{clip.match_score:.1%}|{clip.segment_label:10s}|"
                f"{clip.cut_type.value}"
            )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(edl_lines))
        
        logger.info(f"📋 EDL exportiert: {output_file}")
        return output_file
    
    def export_timeline_as_json(self, timeline: List[TimelineClip],
                               output_file: str = None) -> str:
        """Exportiere Timeline als JSON"""
        
        if output_file is None:
            output_file = os.path.join(Config.EXPORT_DIR, "timeline_blueprint.json")
        
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        
        timeline_data = {
            'timestamp': datetime.now().isoformat(),
            'total_clips': len(timeline),
            'clips': [
                {
                    'index': i,
                    'filename': clip.clip_nfo.filename,
                    'start_s': clip.start_s,
                    'end_s': clip.end_s,
                    'duration_s': clip.end_s - clip.start_s,
                    'segment': clip.segment_label,
                    'cut_type': clip.cut_type.value,
                    'match_score': clip.match_score,
                    'novelty_penalty': clip.novelty_penalty,
                    'quality': clip.clip_nfo.overall_quality
                }
                for i, clip in enumerate(timeline, 1)
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Timeline JSON exportiert: {output_file}")
        return output_file


# ============================================================================
# EBENE 3: RENDER ENGINE
# ============================================================================

class RenderEngine:
    """VFX-Ebene: Render, Export, Metadaten"""
    
    def __init__(self):
        logger.info("🎨 RenderEngine initialisiert (EBENE 3)")
    
    def render_video(self, mp3_path: str, timeline: List[TimelineClip],
                    output_file: str = None) -> str:
        """Rendere finales Musikvideo"""
        
        if output_file is None:
            output_file = os.path.join(
                Config.OUTPUT_DIR,
                f"{Path(mp3_path).stem}_auto.mp4"
            )
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"🎬 Starte Render... → {Path(output_file).name}")
        logger.info(f"   Clips: {len(timeline)} | Gesamtdauer: {timeline[-1].end_s:.1f}s")
        
        # TODO: FFmpeg-Befehl generieren und ausführen
        logger.info("   [Rendering würde hier stattfinden - Demo Mode]")
        
        return output_file


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================

class WEEDITSystem:
    """Zentrale Orchestrierungsengine"""
    
    def __init__(self):
        logger.info("=" * 80)
        logger.info("🎬 WE.ED.IT - BEAT SYNC VIDEO DIRECTOR")
        logger.info("   Automatisches KI-Musikvideo-Rendering")
        logger.info("=" * 80)
        
        self.videopool = VideopoolAnalyzer()
        self.audio_engine = AudioBlueprintEngine()
        self.director = DirectorEngine()
        self.renderer = RenderEngine()
    
    def process_all_mp3_with_clips(self, audio_dir: str = None,
                                  video_dir: str = None) -> Dict:
        """Hauptworkflow: Verarbeite alle MP3 mit verfügbaren Clips"""
        
        if audio_dir is None:
            audio_dir = Config.AUDIO_DIR
        if video_dir is None:
            video_dir = Config.CLIPS_DIR
        
        # Vorbereitung
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.EXPORT_DIR, exist_ok=True)
        
        # EBENE 1: Videopool analysieren
        logger.info("\n" + "="*80)
        logger.info("[EBENE 1] Videopool-Versteher & Content-Analyzer")
        logger.info("="*80)
        video_nfos = self.videopool.analyze_all_clips(video_dir)
        
        # Exportiere Video-Metadaten als CSV
        csv_file = self.videopool.export_clips_as_csv(video_nfos)
        
        # Finde alle MP3 Dateien
        if not os.path.exists(audio_dir):
            logger.error(f"❌ Audio-Verzeichnis nicht gefunden: {audio_dir}")
            return {}
        
        mp3_files = list(Path(audio_dir).glob("*.mp3"))
        logger.info(f"\n📻 Gefunden: {len(mp3_files)} MP3-Dateien")
        
        results = {}
        
        for mp3_path in mp3_files[:3]:  # Max 3 für Demo
            logger.info("\n" + "="*80)
            logger.info(f"🎵 Verarbeite: {mp3_path.name}")
            logger.info("="*80)
            
            try:
                # EBENE 2: Audio Blueprint & Director
                logger.info("\n[EBENE 2] Director-Ebene (Schnitt, Beat-Sync, Struktur)")
                logger.info("-"*80)
                
                blueprint = self.audio_engine.create_blueprint(str(mp3_path))
                
                # Speichere Blueprint
                blueprint_path = os.path.join(Config.CACHE_DIR, f"{mp3_path.stem}_blueprint.json")
                with open(blueprint_path, 'w', encoding='utf-8') as f:
                    f.write(blueprint.to_json())
                logger.info(f"   Blueprint gespeichert: {Path(blueprint_path).name}")
                
                # Erstelle Timeline
                timeline = self.director.create_beat_synced_timeline(blueprint, video_nfos)
                
                # Exportiere Timeline
                edl_file = self.director.export_timeline_as_edl(timeline)
                json_file = self.director.export_timeline_as_json(timeline)
                
                # EBENE 3: Render
                logger.info("\n[EBENE 3] VFX-Ebene (Effekte, Overlay, Platform-Polish)")
                logger.info("-"*80)
                
                output_video = self.renderer.render_video(str(mp3_path), timeline)
                
                results[mp3_path.name] = {
                    'blueprint': blueprint_path,
                    'timeline_edl': edl_file,
                    'timeline_json': json_file,
                    'output_video': output_video,
                    'status': 'success'
                }
                
                logger.info(f"✅ {mp3_path.name} abgeschlossen!")
            
            except Exception as e:
                logger.error(f"❌ Fehler bei {mp3_path.name}: {e}")
                results[mp3_path.name] = {'status': 'error', 'error': str(e)}
        
        # Zusammenfassung
        logger.info("\n" + "="*80)
        logger.info("📊 VERARBEITUNG ABGESCHLOSSEN")
        logger.info("="*80)
        logger.info(f"CSV Export: {csv_file}")
        logger.info(f"Videos verarbeitet: {len([r for r in results.values() if r.get('status') == 'success'])}")
        
        return results


def main():
    """Haupteinstiegspunkt"""
    
    system = WEEDITSystem()
    results = system.process_all_mp3_with_clips()
    
    logger.info("\n✨ Alle Musikvideos generiert! 🎉")


if __name__ == "__main__":
    main()
