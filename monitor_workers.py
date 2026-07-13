#!/usr/bin/env python3
"""
Worker Activity Monitor

This script monitors the activity logs created by lw_synth_1d5D.py workers
to identify which workers are hung and on which pixel/task.

Usage:
    # Check current status of all workers
    python monitor_workers.py
    
    # Show last N lines from each worker
    python monitor_workers.py --tail 20
    
    # Continuous monitoring (updates every 5 seconds)
    python monitor_workers.py --watch
    
    # Check for workers that haven't updated in X seconds
    python monitor_workers.py --timeout 300
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """Parse a worker activity log and extract key information."""
    if not os.path.exists(log_path):
        return None
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Get worker rank from filename
        worker_rank = log_path.stem.split('_')[2]
        
        # Parse last few important lines
        last_activity = None
        current_task = None
        current_pixel = None
        state = "UNKNOWN"
        
        for line in reversed(lines):
            if 'READY signal sent' in line:
                state = "WAITING_FOR_TASK"
                last_activity = line
                break
            elif 'calling synth()' in line:
                state = "COMPUTING"
                last_activity = line
                # Extract task and pixel info
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'Task':
                        current_task = parts[i+1].rstrip(',')
                    if part == 'pixel':
                        current_pixel = parts[i+1].split('/')[0]
                break
            elif 'SUCCESS' in line:
                state = "COMPLETED_PIXEL"
                last_activity = line
                break
            elif 'DONE signal sent' in line:
                state = "COMPLETED_TASK"
                last_activity = line
                break
            elif 'FAILED' in line:
                state = "FAILED"
                last_activity = line
                break
            elif 'Received task' in line:
                state = "RECEIVED_TASK"
                last_activity = line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'task':
                        current_task = parts[i+1].rstrip(',')
                break
        
        if last_activity:
            # Extract timestamp from line
            timestamp_str = last_activity.split('[')[1].split(']')[0]
            last_time = datetime.fromisoformat(timestamp_str)
        else:
            last_time = None
        
        return {
            'rank': worker_rank,
            'state': state,
            'current_task': current_task,
            'current_pixel': current_pixel,
            'last_activity': last_activity.strip() if last_activity else None,
            'last_time': last_time,
            'log_path': log_path,
            'num_lines': len(lines)
        }
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}", file=sys.stderr)
        return None


def find_worker_logs(log_dir="/tmp"):
    """Find all worker activity logs."""
    log_dir = Path(log_dir)
    return sorted(log_dir.glob("lw_worker_*_activity.log"))


def check_worker_status(timeout_seconds=None):
    """Check the status of all workers."""
    log_files = find_worker_logs()
    
    if not log_files:
        print("No worker activity logs found in /tmp/")
        print("Workers may not have started yet, or logging is not enabled.")
        return []
    
    print(f"Found {len(log_files)} worker log(s)\n")
    
    workers_info = []
    now = datetime.now()
    
    for log_file in log_files:
        info = parse_log_file(log_file)
        if info:
            workers_info.append(info)
    
    # Sort by rank
    workers_info.sort(key=lambda x: int(x['rank']))
    
    print(f"{'Rank':<6} {'State':<20} {'Task':<8} {'Pixel':<8} {'Last Activity Age':<20} {'Status'}")
    print("-" * 100)
    
    stuck_workers = []
    
    for info in workers_info:
        rank = info['rank']
        state = info['state']
        task = info['current_task'] or "N/A"
        pixel = info['current_pixel'] or "N/A"
        
        if info['last_time']:
            age = (now - info['last_time']).total_seconds()
            age_str = f"{age:.1f}s ago"
            
            # Determine if stuck
            status = "OK"
            if timeout_seconds and age > timeout_seconds:
                if state == "COMPUTING":
                    status = "⚠️  POSSIBLY HUNG"
                    stuck_workers.append(info)
                elif state == "WAITING_FOR_TASK":
                    status = "⏳ IDLE (long wait)"
        else:
            age_str = "Unknown"
            status = "?"
        
        print(f"{rank:<6} {state:<20} {task:<8} {pixel:<8} {age_str:<20} {status}")
    
    print()
    
    if stuck_workers:
        print(f"\n⚠️  WARNING: {len(stuck_workers)} worker(s) may be hung:\n")
        for info in stuck_workers:
            print(f"  Worker {info['rank']}: stuck on task {info['current_task']}, pixel {info['current_pixel']}")
            print(f"    Last activity: {info['last_activity']}")
            print(f"    Log file: {info['log_path']}\n")
    
    return workers_info


def tail_worker_log(log_path, num_lines=20):
    """Display the last N lines of a worker log."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    print(f"\n{'='*80}")
    print(f"Last {num_lines} lines from {log_path.name}")
    print('='*80)
    for line in lines[-num_lines:]:
        print(line.rstrip())


def watch_workers(interval=5, timeout_seconds=300):
    """Continuously monitor workers."""
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"Worker Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Refreshing every {interval}s, timeout threshold: {timeout_seconds}s\n")
            check_worker_status(timeout_seconds=timeout_seconds)
            print(f"\nPress Ctrl+C to exit...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor MPI worker activity logs to identify hung workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--tail', '-t',
        type=int,
        metavar='N',
        help='Show last N lines from each worker log'
    )
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Continuously monitor workers (refresh every 5s)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        metavar='SECONDS',
        help='Flag workers as possibly hung if no activity for N seconds (default: 300)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        metavar='SECONDS',
        help='Refresh interval for watch mode (default: 5)'
    )
    
    args = parser.parse_args()
    
    if args.watch:
        watch_workers(interval=args.interval, timeout_seconds=args.timeout)
    else:
        workers_info = check_worker_status(timeout_seconds=args.timeout)
        
        if args.tail and workers_info:
            for info in workers_info:
                tail_worker_log(info['log_path'], num_lines=args.tail)


if __name__ == '__main__':
    main()
