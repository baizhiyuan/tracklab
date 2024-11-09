import os
import json

# Path to the video_bzy directory
video_bzy_dir = "/garage#2/projects/data/video_bzy"

# Iterate through subdirectories under video_bzy
for league_dir in os.listdir(video_bzy_dir):
    league_path = os.path.join(video_bzy_dir, league_dir)
    if os.path.isdir(league_path):
        sequences = []
        id_counter = 1  # Start ID counter at 1

        # Collect sequence info for all seasons and matches within the league directory
        for season_dir in os.listdir(league_path):
            season_path = os.path.join(league_path, season_dir)
            if os.path.isdir(season_path):
                for match_dir in os.listdir(season_path):
                    match_path = os.path.join(season_path, match_dir)
                    if os.path.isdir(match_path):
                        img1_path = os.path.join(match_path, "img1")
                        if os.path.exists(img1_path):
                            # Count number of frames (images) in the img1 folder
                            n_frames = len([f for f in os.listdir(img1_path) if f.endswith(".jpg")])
                            sequences.append({
                                "id": id_counter,  # Use unique ID
                                "name": os.path.join(season_dir, match_dir),  # Path without league prefix
                                "n_frames": n_frames
                            })
                            id_counter += 1  # Increment ID counter to ensure uniqueness

        # Create sequences_info.json for the current league directory
        sequences_info = {
            "version": "1.3",
            league_dir: sequences
        }

        # Save to sequences_info.json in the league directory
        json_path = os.path.join(league_path, "sequences_info.json")
        with open(json_path, 'w') as json_file:
            json.dump(sequences_info, json_file, indent=4)

        print(f"Created {json_path}")


# import os
# import json

# # Path to the video_bzy directory
# video_bzy_dir = "/garage#2/projects/data/video_bzy"

# # Iterate through subdirectories under video_bzy
# for league_dir in os.listdir(video_bzy_dir):
#     league_path = os.path.join(video_bzy_dir, league_dir)
#     if os.path.isdir(league_path):
        
#         # Iterate through each season directory under the league
#         for season_dir in os.listdir(league_path):
#             season_path = os.path.join(league_path, season_dir)
#             if os.path.isdir(season_path):
#                 sequences = []

#                 # Collect sequence info for all matches within the season directory
#                 for idx, match_dir in enumerate(os.listdir(season_path)):
#                     match_path = os.path.join(season_path, match_dir)
#                     if os.path.isdir(match_path):
#                         img1_path = os.path.join(match_path, "img1")
#                         if os.path.exists(img1_path):
#                             # Count number of frames (images) in the img1 folder
#                             n_frames = len([f for f in os.listdir(img1_path) if f.endswith(".jpg")])
#                             sequences.append({
#                                 "id": idx,
#                                 "name": match_dir,  # Only the match name
#                                 "n_frames": n_frames
#                             })

#                 # Create sequences_info.json for the current season directory
#                 sequences_info = {
#                     "version": "1.3",
#                     season_dir: sequences
#                 }

#                 # Save to sequences_info.json in the season directory
#                 json_path = os.path.join(season_path, "sequences_info.json")
#                 with open(json_path, 'w') as json_file:
#                     json.dump(sequences_info, json_file, indent=4)

#                 print(f"Created {json_path}")



# import os
# import json

# # Path to the video_bzy directory
# video_bzy_dir = "/garage#2/projects/data/video_bzy"

# # Iterate through subdirectories under video_bzy
# for league_dir in os.listdir(video_bzy_dir):
#     league_path = os.path.join(video_bzy_dir, league_dir)
#     if os.path.isdir(league_path):
#         sequences = []

#         # Collect sequence info for all seasons and matches within the league directory
#         for season_dir in os.listdir(league_path):
#             season_path = os.path.join(league_path, season_dir)
#             if os.path.isdir(season_path):
#                 for idx, match_dir in enumerate(os.listdir(season_path)):
#                     match_path = os.path.join(season_path, match_dir)
#                     if os.path.isdir(match_path):
#                         img1_path = os.path.join(match_path, "img1")
#                         if os.path.exists(img1_path):
#                             # Count number of frames (images) in the img1 folder
#                             n_frames = len([f for f in os.listdir(img1_path) if f.endswith(".jpg")])
#                             sequences.append({
#                                 "id": idx,
#                                 "name": os.path.join(season_dir, match_dir),  # Path without league prefix
#                                 "n_frames": n_frames
#                             })

#         # Create sequences_info.json for the current league directory
#         sequences_info = {
#             "version": "1.3",
#             league_dir: sequences
#         }

#         # Save to sequences_info.json in the league directory
#         json_path = os.path.join(league_path, "sequences_info.json")
#         with open(json_path, 'w') as json_file:
#             json.dump(sequences_info, json_file, indent=4)

#         print(f"Created {json_path}")
