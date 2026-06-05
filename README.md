# 🎬 ART.WE.ED.IT
## Automated Rhythm-Triggered Video Editor with Intelligent Tagging

An advanced AI-powered music video generation system that automatically synchronizes video cuts to audio beats with frame-precision. Combines multimodal LLMs, deterministic audio analysis, and intelligent clip matching.

---

## ✨ Key Features

- **🎵 Precision Audio Analysis** — BeatSync Engine with transient detection, section analysis, and synced lyrics parsing
- **🎥 Semantic Video Classification** — Deterministic + VLM-based analysis (shot size, camera motion, lighting)
- **🧠 Intelligent Clip Matching** — Sentence-transformer embeddings with exponential repetition avoidance
- **⚡ Hardware-Optimized Rendering** — NVIDIA NVENC H.264/H.265, CPU fallback, ProRes support
- **🎛️ VideoVault UI** — Real-time metadata management with bulk editing and reporting
- **📊 Multi-Agent Architecture** — Playwriter, Editor, Reviewer, and Render agents

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (for rendering)
- CUDA 11.8+ (optional, for NVIDIA GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/oidasheim089/WE.ED.IT.git
cd WE.ED.IT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install CUDA-enabled PyTorch for GPU rendering
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```bash
# Generate music video
python art_weedit_engine.py /path/to/audio.mp3 /path/to/video/folder \
  --project "MY_PROJECT" \
  --style "Fast-cut, high-energy, monochrome aesthetic"
```

### Launch VideoVault UI

```bash
# Start metadata management interface
python video_vault_ui.py

# Opens at http://localhost:7860
```

---

## 📋 System Architecture

### Stage 1: 🎵 Audio Analysis (BeatSync Engine)

**5-Stage Pipeline:**

1. **Transient Detection** — STFT-based onset detection with 98th percentile thresholding
2. **Beat Grid Computation** — Librosa beat tracking with stability analysis
3. **Energy Features** — RMS energy, spectral centroid, spectral flux, tempogram
4. **Section Detection** — Chroma-based structural segmentation (intro, verse, chorus, drop, etc.)
5. **Cut Density Determination** — Energy-adaptive cut point selection

**Key Outputs:**
```python
AudioAnalysis(
    bpm=128.0,
    beat_frames=[...],              # Frame indices of beats
    phrase_frames=[...],            # Phrase boundaries
    section_labels=['intro', 'verse', 'chorus', 'drop'],
    synced_lyrics=[...],            # From SYLT ID3 frames
    rms_energy=[...],
    spectral_centroid=[...]
)
```

### Stage 2: 🎥 Video Classification

**Deterministic Analysis (No AI):**
- Scene cut detection (histogram comparison)
- Motion strength estimation (optical flow)
- Focus score (Laplacian variance)
- Beauty score (combination metric)

**Semantic Analysis (Optional VLM):**
- Shot size (extreme_closeup, closeup, medium_shot, wide_shot)
- Camera motion (static, pan, tilt, zoom, handheld, tracking, etc.)
- Lighting (lowkey, highkey, neon, backlighting, etc.)
- Emotional tags (action, combat, chase, explosive, etc.)

**Metadata Cache:**
```
input/video_analysis_cache/
├── clip_001_metadata.json
├── clip_002_metadata.json
└── all_metadata_export.json
```

### Stage 3: 🎯 Semantic Matching with Repetition Avoidance

**Embedding Pipeline:**
1. Audio context (lyrics + song description + style) → 384-dim vector (all-MiniLM-L6-v2)
2. Video metadata (shot, camera, lighting, tags) → 384-dim vector
3. Cosine similarity matching
4. Exponential damping: `similarity_dampened = cosine_sim × e^(-λ×N)` where N = usage count

**Prevents monotone repetition** through dynamic penalty function.

### Stage 4: ⚡ Hardware-Optimized Rendering

```
┌─ Detect GPU
├─ NVIDIA NVENC H.264 (⚡ Fast)
├─ NVIDIA NVENC H.265 (⚡ Smaller files)
├─ CPU H.264 (💾 Universal)
└─ ProRes 422 (✓ Professional)
```

Auto-falls back to CPU rendering if GPU unavailable.

### Stage 5: 🎛️ VideoVault UI

Gradio-based interface for:
- **Individual Tagging** — Manually refine single video metadata
- **Bulk Editing** — Apply changes to multiple videos simultaneously
- **Reporting** — Generate distribution reports
- **Export** — JSON export of all cached metadata

---

## 🔧 Advanced Configuration

### Custom Audio Analysis Parameters

```python
from art_weedit_engine import BeatSyncEngine

engine = BeatSyncEngine(sr=22050)  # Sample rate

# Get detailed audio analysis
audio_data = engine.full_analysis('/path/to/song.mp3')
print(f"Detected BPM: {audio_data.bpm}")
print(f"Sections: {audio_data.section_labels}")
```

### Custom Rendering Parameters

```python
from art_weedit_engine import RenderEngine, RenderMode

render = RenderEngine(output_dir="output")

# Force specific render mode
mode = RenderMode.NVIDIA_H264  # or CPU_H264, PRORES_PROXY

output_file = render.render_video(
    input_clips=[(clip_path, start, end), ...],
    audio_path="song.mp3",
    output_filename="final_video.mp4",
    render_mode=mode
)
```

### Tune Repetition Avoidance

```python
from art_weedit_engine import RepetitionAvoidanceEngine

# Higher lambda = stronger penalty for repeated clips
avoidance = RepetitionAvoidanceEngine(lambda_decay=0.8)

best_clip_id, score = avoidance.find_best_match(
    audio_embedding,
    video_embeddings,
    video_ids,
    alternative_threshold=0.70  # Min similarity to accept
)
```

---

## 📂 Directory Structure

```
WE.ED.IT/
├── ARCHITECTURE.md                 # Full technical specification
├── requirements.txt                # Python dependencies
├── art_weedit_engine.py           # Core orchestration engine
├── video_vault_ui.py              # Metadata management UI
├── input/
│   ├── audio/                     # Place .mp3 files here
│   ├── video/                     # Place .mp4/.mkv files here
│   └── video_analysis_cache/      # Auto-generated metadata
└── output/
    ├── PROJECT_v1.mp4            # Rendered video (v1)
    └── PROJECT_v2.mp4            # Rendered video (v2)
```

---

## 📊 Usage Examples

### Example 1: Basic Music Video Generation

```bash
python art_weedit_engine.py \
  input/audio/song.mp3 \
  input/video \
  --project "SONG_TITLE" \
  --style "Fast-cut, monochrome, neon accents"
```

### Example 2: Custom Audio Analysis

```python
from art_weedit_engine import BeatSyncEngine

engine = BeatSyncEngine()
analysis = engine.full_analysis("song.mp3")

print(f"BPM: {analysis.bpm}")
print(f"Sections: {analysis.section_labels}")

# Extract synced lyrics
for lyric in analysis.synced_lyrics[:5]:
    print(f"{lyric['timestamp_s']:.2f}s: {lyric['text']}")
```

### Example 3: Batch Metadata Management

```bash
# Start VideoVault UI
python video_vault_ui.py

# Then in browser at http://localhost:7860:
# 1. Go to "Bulk Edit" tab
# 2. Enter indices: "0,1,2,3,4"
# 3. Set shot_size: "wide_shot"
# 4. Add tags: "action,fast_cut"
# 5. Click "Apply to All Selected"
```

---

## 🔬 Performance Benchmarks

| Component | Speed | Hardware |
|-----------|-------|----------|
| Audio Analysis | 1-3s per song | CPU |
| Video Classification (Deterministic) | ~10s per hour of footage | CPU |
| Video Classification (Semantic/VLM) | ~1-2min per hour of footage | GPU recommended |
| Embedding Generation | 100-200 clips/sec | CPU |
| Rendering (NVIDIA NVENC) | 5-10x realtime | NVIDIA GPU |
| Rendering (CPU H.264) | 1-2x realtime | Multi-core CPU |

---

## 🛠️ Troubleshooting

### "nvidia-smi not found"
→ Install NVIDIA GPU drivers and CUDA toolkit, or rendering will fall back to CPU

### "ModuleNotFoundError: No module named 'librosa'"
→ Run `pip install -r requirements.txt` again

### VideoVault UI not accessible
→ Ensure port 7860 is open: `http://localhost:7860`

### Rendering produces corrupted video
→ Install FFmpeg: `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux)

---

## 📚 Technical References

### Core Papers & Methods

- **Beat Tracking:** Ellis, D. P. (2007). "Beat tracking by dynamic programming"
- **Onset Detection:** Bello, J. P., et al. (2005). "A tutorial on onset detection in music signals"
- **Semantic Embeddings:** Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **Optical Flow:** Farnebäck, G. (2003). "Two-frame motion estimation based on polynomial expansion"

### Libraries

- **librosa** — Audio feature extraction and beat tracking
- **sentence-transformers** — Semantic embeddings (384-dimensional)
- **torch** — Deep learning framework
- **OpenCV** — Computer vision operations
- **FFmpeg** — Video encoding and rendering
- **Gradio** — Web UI framework

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Multi-GPU rendering support
- [ ] Advanced VFX transitions (cross-dissolve, wipe, etc.)
- [ ] Real-time preview in VideoVault
- [ ] Gender/artist-specific clip filtering
- [ ] Viral scoring metrics and recommendations
- [ ] WebSocket support for live preview

---

## 📄 License

This project is provided as-is for research and creative use.

---

## 👤 Author

**oidasheim089** — Automated Music Video Production System

---

## 🎯 Roadmap

- **v0.2** — Multi-GPU rendering, advanced transitions
- **v0.3** — Real-time WebSocket preview, viral scoring
- **v0.4** — Artist-aware clip filtering, lyrical alignment
- **v1.0** — Production-ready release

---

## ❓ FAQ

**Q: Can I use my own video clips?**
A: Yes! Place them in `input/video/` with any common format (.mp4, .mkv, .mov, etc.)

**Q: What's the minimum video length for good results?**
A: Recommend at least 50-100 short clips (5-15s each) for diverse matching

**Q: Can I export in other formats?**
A: Yes via the `RenderMode` enum (H.264, H.265, ProRes)

**Q: Does it work without a GPU?**
A: Yes, but rendering will be ~5-10x slower. Recommend NVIDIA GPU for real-time workflows.

**Q: How do I extract synced lyrics from my MP3?**
A: Add SYLT ID3 tag to your MP3 using a tag editor like MusicBrainz Picard
