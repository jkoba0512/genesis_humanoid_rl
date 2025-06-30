#!/usr/bin/env python3
"""Monitor training progress and show when videos will be generated."""

import os
import time
import glob
from datetime import datetime, timedelta

def monitor_progression():
    """Monitor the progression training."""
    
    print("=== Robot Learning Progression Monitor ===")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("\nExpected checkpoints:")
    print("  📹 Stage 0: Untrained (0 steps) - Initial video")
    print("  📹 Stage 1: 25,000 steps (~15-20 min)")
    print("  📹 Stage 2: 50,000 steps (~30-40 min)")
    print("  📹 Stage 3: 75,000 steps (~45-60 min)")
    print("  📹 Stage 4: 100,000 steps (~60-75 min)")
    print("\nMonitoring progress...\n")
    
    start_time = time.time()
    checkpoint_dir = "./models/progression_demo/checkpoints"
    video_dir = "./models/progression_demo/progression_videos"
    
    # Track which checkpoints we've seen
    seen_checkpoints = set()
    
    while True:
        try:
            # Check for new checkpoints
            if os.path.exists(checkpoint_dir):
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
                for checkpoint in checkpoints:
                    if checkpoint not in seen_checkpoints:
                        seen_checkpoints.add(checkpoint)
                        elapsed = time.time() - start_time
                        print(f"\n✅ New checkpoint created: {os.path.basename(checkpoint)}")
                        print(f"   Time elapsed: {timedelta(seconds=int(elapsed))}")
            
            # Check for new videos
            if os.path.exists(video_dir):
                videos = glob.glob(os.path.join(video_dir, "stage_*.mp4"))
                print(f"\r📊 Progress: {len(videos)} videos generated | "
                      f"Time: {timedelta(seconds=int(time.time() - start_time))}", end="")
            
            # Check if training is complete
            if os.path.exists("./models/progression_demo/final_progression_model.zip"):
                print("\n\n🎉 Training complete! All videos should be generated.")
                print(f"Total time: {timedelta(seconds=int(time.time() - start_time))}")
                break
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n⏸️ Monitoring stopped by user")
            break
    
    # Show final status
    if os.path.exists(video_dir):
        videos = glob.glob(os.path.join(video_dir, "*.mp4"))
        print(f"\nVideos found in {video_dir}:")
        for video in sorted(videos):
            print(f"  - {os.path.basename(video)}")
    
    print("\n📺 You can watch the training progress on TensorBoard:")
    print("   http://localhost:6007")
    print("\n🎬 Once training completes, run:")
    print("   uv run python scripts/generate_progression_videos.py")
    print("   to create high-quality videos from the checkpoints.")

if __name__ == "__main__":
    monitor_progression()