# Worker Activity Monitoring

This directory contains enhanced worker monitoring to help identify hanging MPI workers in `lw_synth_1d5D.py`.

## What Was Added

### 1. Activity Logging in Workers

Each worker now logs its activity to `/tmp/lw_worker_<rank>_activity.log` with timestamps for:

- Worker startup
- Waiting for tasks (READY signal)
- Task receipt (with task index and grain size)
- **Per-pixel synthesis start** (before calling `synth()`)
- **Per-pixel synthesis completion** (with timing)
- Task completion (with total time)
- Worker shutdown

### 2. Monitoring Script

`monitor_workers.py` - A utility to analyze worker logs and identify stuck workers.

## Usage

### Quick Status Check

```bash
python monitor_workers.py
```

Shows current status of all workers:
- Which state each worker is in (COMPUTING, WAITING_FOR_TASK, etc.)
- Current task and pixel being processed
- Time since last activity
- Warnings for workers that may be hung

### Continuous Monitoring

```bash
python monitor_workers.py --watch
```

Refreshes every 5 seconds to show live worker status. Press Ctrl+C to exit.

### Custom Timeout Threshold

```bash
python monitor_workers.py --timeout 600
```

Flag workers as "POSSIBLY HUNG" if they've been in COMPUTING state for more than 600 seconds (default: 300s).

### View Detailed Logs

```bash
python monitor_workers.py --tail 30
```

Show the last 30 lines from each worker's activity log.

### During MPI Run

In one terminal, run your MPI job:
```bash
mpirun -n 64 python lw_synth_1d5D.py <your_args>
```

In another terminal, monitor the workers:
```bash
python monitor_workers.py --watch --timeout 300
```

## Identifying Hung Workers

When a worker hangs, you'll see:

1. **In the monitor output:**
   ```
   Rank   State                Task     Pixel    Last Activity Age    Status
   ------------------------------------------------------------------------------------
   42     COMPUTING            127      15       482.3s ago           ⚠️  POSSIBLY HUNG
   ```

2. **In the worker activity log** (`/tmp/lw_worker_42_activity.log`):
   ```
   [2026-07-13 14:23:15.123456] Task 127, pixel 15/32 - calling synth()
   [no further activity for 8+ minutes]
   ```

3. **In stdout** (from the MPI job):
   ```
   [Worker 42] Starting task 127 with 32 pixels
   [no further output]
   ```

This tells you:
- **Worker 42** is stuck
- **Task 127, pixel 15** is the problematic one
- The hang occurs inside the `synth()` call (non-converging NLTE solver or ray tracing)

## Diagnosing the Problem

Once you identify a stuck pixel, check:

1. **Atmospheric parameters** - The worker logs the parameters when exceptions occur, but for hangs (infinite loops), you'll need to identify the pixel from the task index and pixel number.

2. **Extract the problematic atmosphere:**
   ```python
   # If task 127, pixel 15 is stuck, and grain_size=32:
   task_index = 127
   pixel_in_task = 15
   global_pixel_index = task_index * 32 + pixel_in_task
   
   # Load your atmosphere array and examine:
   # atmosarr['T'][global_pixel_index]
   # atmosarr['Pg'][global_pixel_index]
   # etc.
   ```

3. **Common causes:**
   - Extreme temperature inversions
   - Very low or high gas pressure
   - Unphysical velocity fields
   - Non-converging NLTE iteration in specific atmospheric conditions

## Next Steps: Adding Timeouts

The monitoring helps **identify** where workers hang, but doesn't **prevent** hangs. To prevent indefinite hangs, consider:

1. **Per-pixel timeout** using signal.alarm() (Unix only)
2. **Max iteration limits** in the NLTE solver
3. **Convergence checks** to detect non-converging cases early
4. **Pre-validation** of atmospheric parameters before synthesis

See the `MPI Hang Debugger` agent for help implementing these:
```
@mpi-hang-debugger How do I add a timeout to the synth() call in lw_synth_1d5D.py?
```

## Log Files

All logs are in `/tmp/`:
- `/tmp/lw_worker_<rank>_activity.log` - Activity/progress log (new)
- `/tmp/lw_worker_<rank>_errors.log` - Exception/error details (existing)
- `/tmp/lw_failed_pixel_r<rank>_t<task>_p<pixel>.pkl` - Failed pixel data (existing)

## Cleaning Up Logs

Before a new run:
```bash
rm /tmp/lw_worker_*_activity.log /tmp/lw_worker_*_errors.log
```

Or keep old logs for comparison:
```bash
mkdir -p old_logs
mv /tmp/lw_worker_*.log old_logs/run_$(date +%Y%m%d_%H%M%S)/
```
