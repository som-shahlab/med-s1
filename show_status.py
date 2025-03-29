#!/usr/bin/env python3

import json
import datetime
import pytz
import os
from pathlib import Path

def load_results():
    try:
        with open(Path(__file__).parent / "results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"experiments": {}}

def get_time_str(timestamp_str):
    if not timestamp_str:
        return " " * 14  # Width of "MM/DD HH:MMx√"
    
    try:
        dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        pst = pytz.timezone('America/Los_Angeles')
        dt_pst = dt.astimezone(pst)
        return dt_pst.strftime("%-m/%-d %-I:%M%p").lower()
    except:
        return " " * 14

def get_status_str(phase_data):
    return " "  # Default to empty string
    if not phase_data:
        return " "  # Not run
    
    success = phase_data.get("success", False)
    return "✓" if success else "✗"

def format_experiment_line(name, results, width):
    # Get status for each phase
    phases = ["curation", "training", "eval"]
    times = []
    statuses = []
    
    for phase in phases:
        phase_data = results.get(phase, {})
        time_str = get_time_str(phase_data.get("timestamp", ""))
        status_str = get_status_str(phase_data)
        times.append(time_str)
        statuses.append(status_str)
    
    # Format the line with fixed widths
    name_width = width - (16 * 3)  # Each phase takes 16 chars (time + status)
    line = f"{name[:name_width]:<{name_width}}"
    
    for i in range(3):
        if times[i].strip():  # If there's a timestamp
            line += f" {times[i]}{statuses[i]}"
        else:
            line += " " * 16  # Empty space for unused phase
    
    return line

def main():
    data = load_results()
    
    # Get terminal width
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 100  # fallback width
    
    # Define experiment groups
    groups = {
        "Extractions": [
            "medqa-1k-random",
            "medqa-1k-random-step-extract",
            "medqa-1k-random-evidence-extract",
            "medqa-1k-random-markdown-extract",
            "medqa-1k-random-list-extract",
            "medqa-1k-random-note-extract",
            "medqa-1k-random-qa-extract",
            "medqa-1k-random-socratic-extract",
            "medqa-1k-random-decision-tree-extract"
        ],
        "Perturbations": [
            "medqa-1k-random-collapse-33",
            "medqa-1k-random-collapse-66",
            "medqa-1k-random-collapse-100",
            "medqa-1k-random-skip-33",
            "medqa-1k-random-skip-66",
            "medqa-1k-random-skip-100",
            "medqa-1k-random-shuffle-33",
            "medqa-1k-random-shuffle-66",
            "medqa-1k-random-shuffle-100",
            "medqa-1k-random-irrelevant-33",
            "medqa-1k-random-irrelevant-66",
            "medqa-1k-random-irrelevant-100",
            "medqa-1k-random-wrong-answer-33",
            "medqa-1k-random-wrong-answer-66",
            "medqa-1k-random-wrong-answer-100"
        ],
        "Restorations": [
            "medqa-1k-random-collapse-33-restore",
            "medqa-1k-random-collapse-66-restore",
            "medqa-1k-random-collapse-100-restore",
            "medqa-1k-random-skip-33-restore",
            "medqa-1k-random-skip-66-restore",
            "medqa-1k-random-skip-100-restore",
            "medqa-1k-random-shuffle-33-restore",
            "medqa-1k-random-shuffle-66-restore",
            "medqa-1k-random-shuffle-100-restore",
            "medqa-1k-random-irrelevant-33-restore",
            "medqa-1k-random-irrelevant-66-restore",
            "medqa-1k-random-irrelevant-100-restore"
        ]
    }
    
    # Print header
    print("\nExperiment Status (✓=success, ✗=failed, blank=not run)")
    print(f"{'Name':<{width-48}} {'Curation':<16} {'Training':<16} {'Eval':<16}")
    print("-" * width)
    
    # Print each group
    for group_name, experiments in groups.items():
        print(f"\n{group_name}:")
        for exp in experiments:
            exp_data = data["experiments"].get(exp, {})
            results = exp_data.get("results", {})
            print(format_experiment_line(exp, results, width))

if __name__ == "__main__":
    main()