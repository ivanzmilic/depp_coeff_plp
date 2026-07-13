#!/usr/bin/env python3
"""
Test script to verify worker activity logging works.
Creates fake worker logs to test the monitor.
"""

import os
from datetime import datetime, timedelta

def create_test_log(rank, state="computing", task=42, pixel=15, age_seconds=0):
    """Create a test worker activity log."""
    log_path = f"/tmp/lw_worker_{rank}_activity.log"
    
    base_time = datetime.now() - timedelta(seconds=age_seconds)
    
    with open(log_path, 'w') as f:
        f.write(f"Worker {rank} started at {base_time}\n")
        f.write(f"[{base_time}] READY signal sent, waiting for task...\n")
        
        base_time += timedelta(seconds=1)
        f.write(f"[{base_time}] Received task {task}, grain_size=32, stokes=False\n")
        
        if state == "computing":
            base_time += timedelta(seconds=2)
            f.write(f"[{base_time}] Task {task}, pixel {pixel}/32 - starting synthesis\n")
            base_time += timedelta(seconds=1)
            f.write(f"[{base_time}] Task {task}, pixel {pixel}/32 - calling synth()\n")
        
        elif state == "completed":
            for p in range(pixel + 1):
                base_time += timedelta(seconds=2)
                f.write(f"[{base_time}] Task {task}, pixel {p}/32 - calling synth()\n")
                base_time += timedelta(seconds=10)
                f.write(f"[{base_time}] Task {task}, pixel {p}/32 - SUCCESS in 10.23s\n")
            
            base_time += timedelta(seconds=1)
            f.write(f"[{base_time}] Task {task} COMPLETED in 320.45s (10.01s per pixel)\n")
            f.write(f"[{base_time}] DONE signal sent for task {task}\n")
        
        elif state == "waiting":
            base_time += timedelta(seconds=1)
            f.write(f"[{base_time}] DONE signal sent for task {task-1}\n")
            base_time += timedelta(seconds=1)
            f.write(f"[{base_time}] READY signal sent, waiting for task...\n")
    
    print(f"Created test log for worker {rank} ({state}, age={age_seconds}s)")

# Create test scenarios
print("Creating test worker logs in /tmp/\n")

# Worker 0: normal, just completed a task
create_test_log(0, state="completed", task=10, pixel=31, age_seconds=5)

# Worker 1: computing, been working for 30 seconds (OK)
create_test_log(1, state="computing", task=15, pixel=8, age_seconds=30)

# Worker 2: computing, been stuck for 400 seconds (HUNG!)
create_test_log(2, state="computing", task=23, pixel=12, age_seconds=400)

# Worker 3: waiting for task for a long time
create_test_log(3, state="waiting", task=18, pixel=0, age_seconds=250)

# Worker 4: computing, been working for 150 seconds (borderline)
create_test_log(4, state="computing", task=27, pixel=5, age_seconds=150)

print("\nTest logs created! Now run:")
print("  python monitor_workers.py")
print("or:")
print("  python monitor_workers.py --timeout 300")
print("\nExpected results:")
print("  - Worker 0: OK (recently completed)")
print("  - Worker 1: OK (computing, < 300s)")
print("  - Worker 2: POSSIBLY HUNG (computing, > 300s)")
print("  - Worker 3: IDLE (waiting for task)")
print("  - Worker 4: OK (computing, < 300s)")
print("\nClean up with:")
print("  rm /tmp/lw_worker_*_activity.log")
