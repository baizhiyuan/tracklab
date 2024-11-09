import os
import json

# Path to the video_bzy directory
video_bzy_dir = "/garage#2/projects/data/video_bzy"

# Summary of all sequences_info.json paths
summary = []

# Iterate through subdirectories under video_bzy
for league_dir in os.listdir(video_bzy_dir):
    league_path = os.path.join(video_bzy_dir, league_dir)
    if os.path.isdir(league_path):
        for season_dir in os.listdir(league_path):
            season_path = os.path.join(league_path, season_dir)
            if os.path.isdir(season_path):
                # Collect sequence info for all folders within the season directory
                sequences = []
                for idx, match_dir in enumerate(os.listdir(season_path)):
                    match_path = os.path.join(season_path, match_dir)
                    if os.path.isdir(match_path):
                        img1_path = os.path.join(match_path, "img1")
                        if os.path.exists(img1_path):
                            # Count number of frames (images) in the img1 folder
                            n_frames = len([f for f in os.listdir(img1_path) if f.endswith(".jpg")])
                            sequences.append({
                                "id": idx,
                                "name": match_dir,
                                "n_frames": n_frames
                            })

                # Create sequences_info.json
                sequences_info = {
                    "version": "1.3",
                    "sequences": sequences
                }

                # Save to sequences_info.json in the season directory
                json_path = os.path.join(season_path, "sequences_info.json")
                with open(json_path, 'w') as json_file:
                    json.dump(sequences_info, json_file, indent=4)

                # Add to summary
                summary.append({
                    "league": league_dir,
                    "season": season_dir,
                    "path": json_path
                })

                print(f"Created {json_path}")

# Create a summary JSON file at the root of video_bzy
summary_path = os.path.join(video_bzy_dir, "sequences_summary.json")
with open(summary_path, 'w') as summary_file:
    json.dump(summary, summary_file, indent=4)

print(f"Created summary file at {summary_path}")
