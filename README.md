# Portrait Converter

Convert landscape podcast videos to portrait format (9:16) by intelligently focusing on the speaking person.

## Features

- **Scene Detection**: Automatically splits video into scenes using content-based detection
- **YOLO-11 Pose Estimation**: Detects persons with 80%+ confidence and extracts keypoints
- **RetinaFace Detection**: High-precision face detection with 95%+ confidence
- **Lip Movement Analysis**: Identifies who is speaking based on mouth movement variance
- **Static Crop Calculation**: Ensures eyes/nose stay in frame throughout entire scene
- **GPU Acceleration**: CUDA support for faster processing
- **Debug Visualizations**: Save annotated frames showing detections and crop regions
- **Audio Preservation**: Automatically copies audio from original video

## Quick Start

### Installation

```bash
# Automated setup (recommended)
pip install -r requirements.txt

# Install FFmpeg
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```


**Basic Usage**:
```bash
source venv/bin/activate
python main.py --input landscape.mp4 --output portrait.mp4
deactivate
```

**Debug mode** saves frames to `debug_frames/` showing detections (green=persons, magenta=faces, yellow=crop)

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input landscape video path | Required |
| `--output`, `-o` | Output portrait video path | Required |
| `--debug` | Enable debug mode (saves detection frames) | False |
| `--pose-threshold` | YOLO pose detection threshold (0.0-1.0) | 0.80 |
| `--face-threshold` | RetinaFace detection threshold (0.0-1.0) | 0.95 |
| `--scene-threshold` | Scene detection sensitivity (higher = fewer scenes) | 27.0 |
| `--output-width` | Output video width (pixels) | 1080 |
| `--output-height` | Output video height (pixels) | 1920 |

### Examples

**Process with lower thresholds (detect more people/faces):**
```bash
python main.py --input video.mp4 --output portrait.mp4 \
  --pose-threshold 0.70 --face-threshold 0.90
```

**Generate lower resolution output (faster):**
```bash
python main.py --input video.mp4 --output portrait_720p.mp4 \
  --output-width 720 --output-height 1280
```

**Adjust scene detection:**
```bash
# More scenes (more sensitive to changes)
python main.py --input video.mp4 --output portrait.mp4 --scene-threshold 20.0

# Fewer scenes (less sensitive)
python main.py --input video.mp4 --output portrait.mp4 --scene-threshold 35.0
```

## How It Works

### Processing Pipeline

```
Input Video (Landscape 16:9)
    ↓
[1] Scene Detection
    ↓
[2] For Each Scene:
    ↓
    [2a] YOLO-11 Pose Estimation (detect all persons)
    ↓
    [2b] Count persons:
         - 1 person → Go to [3a]
         - 2+ persons → Go to [2c]
    ↓
    [2c] RetinaFace Detection (detect faces)
    ↓
    [2d] Match poses with faces (verify eyes/nose in face bbox)
    ↓
    [2e] Lip Movement Analysis (find who's talking)
    ↓
    [2f] Select target person:
         - Talking person (if detected)
         - Highest confidence person (fallback)
    ↓
[3] Calculate Static Crop
    ↓
    [3a] Analyze all frames to find bounding box containing all eye/nose positions
    ↓
    [3b] Calculate 9:16 crop region with eyes in upper 30% of frame
    ↓
    [3c] Apply crop to all frames (static crop - no tracking)
    ↓
[4] Stitch All Portrait Scenes
    ↓
[5] Copy Audio from Original
    ↓
Output Video (Portrait 9:16)
```

### Detection Logic

#### Single Person Scene
1. Detect person using YOLO-11 with pose threshold ≥ 0.80
2. Extract eye and nose keypoints from all frames
3. Calculate static crop region that keeps eyes/nose in frame
4. Crop with eyes positioned ~30% from top

#### Multiple Person Scene
1. Detect all persons using YOLO-11
2. Detect faces using RetinaFace with confidence ≥ 0.95
3. Match persons to faces by checking if pose keypoints (eyes, nose) fall within face bounding boxes
4. For each confirmed person, analyze lip movement:
   - Extract mouth region across frames
   - Calculate pixel variance (higher = more movement)
   - Person with highest variance is talking
5. If lip movement is inconclusive, use person with highest YOLO confidence
6. Calculate static crop for target person

#### Fallback Strategies
- **No faces detected**: Use person with highest YOLO confidence
- **No lip movement detected**: Use first person with confirmed face
- **No persons detected**: Skip scene (log warning)

## Project Structure

```
PortraitConverter/
├── main.py                      # Command-line interface
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── INSTALLATION.md              # Installation guide
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Main conversion pipeline
│   ├── models/
│   │   ├── pose_estimator.py    # YOLO-11 pose estimation
│   │   └── face_detector.py     # RetinaFace + lip movement
│   ├── processors/
│   │   ├── scene_detector.py    # Scene detection & splitting
│   │   ├── crop_calculator.py   # Static crop calculation
│   │   └── video_stitcher.py    # Video stitching & audio
│   └── utils/
│       ├── config.py            # Configuration
│       ├── logger.py            # Logging
│       └── visualizer.py        # Debug visualizations
├── output/                      # Output videos (gitignored)
├── debug_frames/                # Debug frames (gitignored)
└── temp_work/                   # Temporary files (gitignored)
```

## Programmatic Usage

```python
from src.pipeline import PortraitConverter
from src.utils.config import Config

# Configure (optional)
Config.set_thresholds(pose_threshold=0.80, face_threshold=0.95)
Config.OUTPUT_WIDTH = 1080
Config.OUTPUT_HEIGHT = 1920

# Create converter
converter = PortraitConverter(debug_mode=True)

# Convert
success = converter.convert("input.mp4", "output.mp4")
```


## Limitations

1. **Static Crop**: The crop region is calculated once per scene and applied to all frames. No dynamic tracking/smoothing.
2. **Lip Movement**: Works best with clear frontal face views. May struggle with:
   - Side profiles
   - Low video quality
   - Heavy compression
3. **Scene Detection**: May over/under-segment in videos with:
   - Gradual lighting changes
   - Frequent cuts
   - Screen recordings with static content
4. **Multi-person Accuracy**: Best with 2-3 people. May struggle with:
   - Large groups (5+ people)
   - People entering/exiting frame frequently
   - Overlapping faces

## Troubleshooting

### Common Issues

**Issue**: Crop focuses on wrong person
- **Solution**: Lower face threshold: `--face-threshold 0.90`
- **Solution**: Enable debug mode to see detections: `--debug`

**Issue**: Eyes/nose go out of frame
- **Solution**: The static crop calculation should prevent this. If it happens, check debug frames to see detection quality.

**Issue**: Too many/few scenes
- **Solution**: Adjust `--scene-threshold` (default: 20.0)
  - Higher = fewer scenes
  - Lower = more scenes

**Issue**: Processing is slow
- **Solution**: Check GPU usage: `nvidia-smi`
- **Solution**: Reduce output resolution: `--output-width 720 --output-height 1280`
- **Solution**: Use CPU for debugging, GPU for final render

**Issue**: "CUDA out of memory"
- **Solution**: Use smaller output resolution
- **Solution**: Close other GPU applications

See [INSTALLATION.md](INSTALLATION.md) for more troubleshooting.

## Requirements

- Python 3.8+
- FFmpeg (for audio handling)
- 8GB+ RAM
- NVIDIA GPU with CUDA (optional, but recommended)

See [requirements.txt](requirements.txt) for Python dependencies.

