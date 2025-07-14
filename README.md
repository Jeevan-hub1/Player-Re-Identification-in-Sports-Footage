# Player Re-identification in Single Feed

A comprehensive solution for tracking and re-identifying players in sports footage using computer vision techniques.

## Overview

This project implements a player re-identification system that maintains consistent player IDs throughout a video, even when players go out of frame and re-enter. The system uses YOLOv11 for player detection and combines multiple feature extraction methods for robust re-identification.

## Features

- **Multi-modal Feature Extraction**: Combines color histograms, shape features, and spatial information
- **Robust Tracking**: Handles occlusion, similar appearances, and camera movement
- **Real-time Processing**: Optimized for efficient video processing
- **Trajectory Smoothing**: Uses player movement history for improved accuracy
- **Comprehensive Output**: Provides both JSON results and annotated video output

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Jeevan-hub1/Player-Re-Identification-in-Sports-Footage>
   cd player-reid
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the provided YOLOv11 model**
   - Download from: https://drive.google.com/file/d/1-5fOSHOSBOUXVP_enOoZNAMScrePVcMD/view
   - Place it in the `models/` directory

## Usage

### Basic Usage

```bash
python player_reid.py --video path/to/15sec_input_720p.mp4 --model path/to/best.pt
```

### Complete Usage with Output

```bash
python player_reid.py \
    --video path/to/15sec_input_720p.mp4 \
    --model path/to/best.pt \
    --output_video output_with_tracking.mp4 \
    --output_json tracking_results.json \
    --confidence 0.5
```

### Command Line Arguments

- `--video`: Path to input video file (required)
- `--model`: Path to YOLOv11 model file (required)
- `--output_video`: Path to save annotated video output (optional)
- `--output_json`: Path to save JSON tracking results (optional)
- `--confidence`: Detection confidence threshold (default: 0.5)

### Example

```bash
python player_reid.py \
    --video data/15sec_input_720p.mp4 \
    --model models/best.pt \
    --output_video results/tracked_video.mp4 \
    --output_json results/tracking_results.json \
    --confidence 0.6
```

## Architecture

### System Components

1. **PlayerFeatureExtractor**: Extracts multi-modal features from player detections
2. **PlayerTracker**: Manages player tracks and handles re-identification
3. **PlayerReID**: Main orchestrator class that combines detection and tracking

### Feature Extraction

The system extracts three types of features:

1. **Color Features**: HSV color histograms for jersey identification
2. **Shape Features**: Aspect ratio, contour properties, and spatial moments
3. **Spatial Features**: Bounding box position and movement patterns

### Tracking Algorithm

1. **Detection**: YOLOv11 detects players in each frame
2. **Feature Extraction**: Multi-modal features extracted from each detection
3. **Similarity Matching**: Cosine similarity used for feature comparison
4. **Assignment**: Greedy assignment algorithm matches detections to existing tracks
5. **Re-identification**: Missing players are re-identified when they reappear

## Output Format

### JSON Results Structure

```json
{
  "frame_number": 0,
  "detections": [
    {
      "player_id": 1,
      "bbox": [x, y, width, height],
      "confidence": 0.95
    }
  ]
}
```

### Video Output

The annotated video includes:
- Colored bounding boxes around each player
- Player ID labels
- Confidence scores
- Consistent colors for each player ID

## Configuration

### Key Parameters

- `confidence_threshold`: Minimum detection confidence (default: 0.5)
- `similarity_threshold`: Minimum similarity for re-identification (default: 0.5)
- `max_disappeared`: Maximum frames a player can be missing (default: 15)
- `color_bins`: Number of bins for color histograms (default: 32)

### Feature Weights

The similarity calculation uses weighted combination:
- Color similarity: 50%
- Shape similarity: 30%
- Spatial similarity: 20%

## Performance Optimization

### Speed Optimizations

1. **Efficient Feature Extraction**: Optimized histogram calculations
2. **Trajectory Prediction**: Reduces search space for re-identification
3. **Feature History**: Uses moving averages for stable matching
4. **Memory Management**: Bounded queues for history storage

### Memory Usage

- Player history limited to 20 frames
- Feature history limited to 5 recent features
- Automatic cleanup of disappeared players

## Challenges and Solutions

### Challenge 1: Similar Player Appearances
**Solution**: Multi-modal feature combination with weighted similarity

### Challenge 2: Players Leaving and Re-entering Frame
**Solution**: Disappeared player tracking with feature-based re-identification

### Challenge 3: Occlusion and Partial Visibility
**Solution**: Trajectory prediction and temporal consistency

### Challenge 4: Camera Movement
**Solution**: Relative spatial features and movement-based matching

## Evaluation Metrics

The system tracks:
- **Identity Switches**: How often player IDs change incorrectly
- **Fragmentation**: How often continuous tracks are broken
- **Detection Accuracy**: Percentage of correctly detected players
- **Re-identification Accuracy**: Success rate of re-identifying returning players

## Limitations

1. **Similar Jerseys**: May struggle with players wearing identical uniforms
2. **Extreme Occlusion**: Long-term occlusion may cause ID switches
3. **Camera Quality**: Low-resolution footage affects feature extraction
4. **Crowded Scenes**: Many overlapping players can reduce accuracy

## Future Improvements

1. **Deep Learning Features**: Integrate CNN-based feature extraction
2. **Pose Estimation**: Use player pose for additional identification cues
3. **Team Classification**: Automatic team detection and separation
4. **Kalman Filtering**: Advanced trajectory prediction
5. **Online Learning**: Adaptive feature updates during tracking

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Solution: Ensure YOLOv11 model path is correct and accessible
   ```

2. **Low Detection Accuracy**
   ```
   Solution: Adjust confidence threshold or check video quality
   ```

3. **Frequent ID Switches**
   ```
   Solution: Increase similarity threshold or adjust feature weights
   ```

4. **Memory Issues**
   ```
   Solution: Reduce history buffer sizes or process shorter video segments
   ```

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for faster processing
2. **Video Resolution**: Higher resolution improves feature extraction quality
3. **Frame Rate**: Higher FPS provides better temporal consistency
4. **Lighting Conditions**: Consistent lighting improves color-based matching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is developed for educational purposes as part of an AI internship assignment.

## Contact

For questions or issues, please contact:
- Email: nandakumarponnala@gmail.com
- GitHub: Jeevan-hub1

## Acknowledgments

- Ultralytics for the YOLOv11 model
- OpenCV community for computer vision tools
- Liat.ai for the assignment opportunity
