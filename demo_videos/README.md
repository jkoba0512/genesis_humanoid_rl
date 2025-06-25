# Demo Videos 🎬

This directory contains demonstration videos of the humanoid robot training and walking.

## Available Videos

### Training Demonstrations
- `genesis_humanoid_rl_demo_YYYYMMDD_HHMMSS.mp4` - Training session recordings

## Creating New Demo Videos

Generate a new demo video:

```bash
# Create a demonstration video
uv run python create_demo_video.py
```

The video will show:
- Robot loading and initialization
- Grounding system in action  
- Joint movement demonstrations
- Observation system capabilities
- Physics simulation quality

## Video Specifications

- **Resolution**: 1280x720 (HD)
- **Frame Rate**: 60 FPS
- **Duration**: ~15 seconds
- **Format**: MP4 (H.264)

## Note

Large video files (>100MB) are excluded from git repository.
For sharing large videos, consider using:
- Git LFS (Large File Storage)
- External hosting (YouTube, Vimeo)
- Cloud storage links

## Evaluation Videos

To record videos of trained models:

```bash
# Record evaluation with rendering
uv run python scripts/evaluate_sb3.py ./models/sb3/best_model --render --episodes 3
```