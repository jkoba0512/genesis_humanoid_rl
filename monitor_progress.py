#!/usr/bin/env python3
"""Simple progress monitor for learning evolution."""

import json
import time
from datetime import datetime

print("🤖 Genesis Humanoid Learning Progress Monitor")
print("=" * 50)

while True:
    try:
        with open('evolution_status.json', 'r') as f:
            status = json.load(f)
        
        print(f"\r📊 Progress: {status['progress']:.1f}% "
              f"({status['current_steps']:,}/{status['total_steps']:,} steps) "
              f"| Last update: {datetime.now().strftime('%H:%M:%S')}", end="", flush=True)
        
        # Check for checkpoints
        if status['current_steps'] in status['checkpoint_steps'][1:]:
            print(f"\n✅ Checkpoint saved at {status['current_steps']:,} steps!")
            
    except:
        pass
    
    time.sleep(5)