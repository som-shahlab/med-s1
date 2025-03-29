#!/usr/bin/env python3

import json
import datetime
import pytz
import os
from pathlib import Path

def load_results():
    with open(Path(__file__).parent / "results.json") as f:
        return json.load(f)

def get_pst_time(timestamp_str):
    if not timestamp_str:
        return ""
    # Parse the timestamp
    dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    # Convert to PST
    pst = pytz.timezone('America/Los_Angeles')
    dt_pst = dt.astimezone(pst)
    
    # Get current time in PST for comparison
    now = datetime.datetime.now(datetime.timezone.utc).astimezone(pst)
    
    # Format based on how recent the timestamp is
    if dt_pst.date() == now.date():
        # Today: just show time
        return f"Today {dt_pst.strftime('%-I:%M%p').lower()}"
    elif dt_pst.date() == now.date() - datetime.timedelta(days=1):
        # Yesterday
        return f"Yesterday {dt_pst.strftime('%-I:%M%p').lower()}"
    else:
        # Other dates
        return dt_pst.strftime("%-m/%-d %-I:%M%p").lower()

def parse_timestamp(timestamp_str):
    """Parse timestamp string to UTC datetime object"""
    if not timestamp_str:
        return None
    # Handle both Z and +00:00 UTC indicators
    dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    # Ensure timezone awareness
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt

def get_column_widths():
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 120  # fallback width
    
    # Calculate widths based on terminal size
    exp_width = min(40, max(30, terminal_width // 3))
    time_width = min(20, max(15, (terminal_width - exp_width) // 4))
    
    return exp_width, time_width

def main():
    # Load results
    data = load_results()
    
    # Get current time in UTC
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(hours=12)
    
    # Track the latest activity for each experiment
    recent_experiments = []
    
    for exp_name, exp_data in data["experiments"].items():
        results = exp_data.get("results", {})
        
        # Get timestamps for each phase
        timestamps = {
            "curation": None,
            "training": None,
            "eval": None
        }
        
        latest_time = None
        
        for phase in ["curation", "training", "eval"]:
            if results.get(phase) and results[phase] and "timestamp" in results[phase]:
                timestamp_str = results[phase]["timestamp"]
                timestamp = parse_timestamp(timestamp_str)
                timestamps[phase] = timestamp_str
                
                if not latest_time or (timestamp and timestamp > latest_time):
                    latest_time = timestamp
        
        # If we have any activity and it's within last 12 hours
        if latest_time and latest_time > cutoff:
            recent_experiments.append({
                "name": exp_name,
                "latest": latest_time,
                "timestamps": timestamps
            })
    
    # Sort by most recent activity
    recent_experiments.sort(key=lambda x: x["latest"], reverse=True)
    
    # Get column widths
    exp_width, time_width = get_column_widths()
    total_width = exp_width + 4 * time_width + 4  # Add some padding
    
    # Print results
    print("\nExperiments with activity in the last 12 hours:\n")
    print(f"{'Experiment':<{exp_width}} {'Recent':<{time_width}} {'Curation':<{time_width}} {'Training':<{time_width}} {'Eval':<{time_width}}")
    print("-" * total_width)
    
    for exp in recent_experiments:
        print(f"{exp['name'][:exp_width]:<{exp_width}} "
              f"{get_pst_time(exp['latest'].isoformat()):<{time_width}} "
              f"{get_pst_time(exp['timestamps']['curation']):<{time_width}} "
              f"{get_pst_time(exp['timestamps']['training']):<{time_width}} "
              f"{get_pst_time(exp['timestamps']['eval']):<{time_width}}")

if __name__ == "__main__":
    main()