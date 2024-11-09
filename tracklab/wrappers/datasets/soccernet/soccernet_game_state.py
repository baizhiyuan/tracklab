import logging
import os
import time
import zipfile
import numpy as np
import pandas as pd
import json

from pathlib import Path
from rich import print
from SoccerNet.Downloader import SoccerNetDownloader
from rich.prompt import Confirm
from tracklab.datastruct import TrackingDataset, TrackingSet
from tracklab.utils import xywh_to_ltwh
from tracklab.utils.progress import progress
from multiprocessing import Pool

log = logging.getLogger(__name__)


class SoccerNetGameState(TrackingDataset):
    def __init__(self,
                 dataset_path: str,
                 nvid: int = -1,
                 vids_dict: dict = None,
                 *args, **kwargs):
        self.dataset_path = Path(dataset_path)
        # if not self.dataset_path.exists():
        #     download_dataset(self.dataset_path)
        # assert self.dataset_path.exists(), f"'{self.dataset_path}' directory does not exist. Please check the path or download the dataset."

        sets = {}
        # Iterate over the leagues specified in vids_dict or eval_set
        if vids_dict is None or len(vids_dict) == 0:
            leagues = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        else:
            leagues = vids_dict.keys()

        for league in leagues:
            league_path = self.dataset_path / league
            if not league_path.exists():
                continue
            sets[league] = load_set(str(league_path), nvid, vids_dict.get(league))

        # Initialize TrackingDataset with the sets
        super().__init__(dataset_path, sets, nvid=-1, vids_dict=None, *args, **kwargs)

    def process_trackeval_results(self, results, dataset_config, eval_config):
        combined_results = results['SUMMARIES']['cls_comb_det_av']
        combined_results['GS-HOTA'] = combined_results.pop('HOTA')
        # In all keys, replace the substring "HOTA" with "GS-HOTA"
        combined_results['GS-HOTA'] = {k.replace('HOTA', 'GS-HOTA'): v for k, v in
                                       combined_results['GS-HOTA'].items()}
        log.info(f"SoccerNet Game State Reconstruction performance GS-HOTA = {combined_results['GS-HOTA']['GS-HOTA']}% (config: EVAL_SPACE={dataset_config['EVAL_SPACE']}, USE_JERSEY_NUMBERS={dataset_config['USE_JERSEY_NUMBERS']}, USE_TEAMS={dataset_config['USE_TEAMS']}, USE_ROLES={dataset_config['USE_ROLES']}, EVAL_DIST_TOL={dataset_config['EVAL_DIST_TOL']})")
        log.info(f"Have a look at 'tracklab/tracklab/configs/dataset/soccernet_gs.yaml' for more details about the GS-HOTA metric and the evaluation configuration.")
        return combined_results

    def save_for_eval(self,
                      detections: pd.DataFrame,
                      image_metadatas: pd.DataFrame,
                      video_metadatas: pd.DataFrame,
                      save_folder: str,
                      bbox_column_for_eval="bbox_ltwh",
                      save_classes=False,
                      is_ground_truth=False,
                      save_zip=True
                      ):
        if is_ground_truth:
            return
        save_path = Path(save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        detections = self.soccernet_encoding(detections.copy(), supercategory="object")
        camera_metadata = self.soccernet_encoding(image_metadatas.copy(), supercategory="camera")
        pitch_metadata = self.soccernet_encoding(image_metadatas.copy(), supercategory="pitch")
        predictions = pd.concat([detections, camera_metadata, pitch_metadata], ignore_index=True)
        zf_save_path = save_path.parents[1] / f"{save_path.parent.name}.zip"
        for id, video in video_metadatas.iterrows():
            file_path = save_path / f"{video['name']}.json"
            video_predictions_df = predictions[predictions["video_id"] == str(id)].copy()
            if not video_predictions_df.empty:
                video_predictions_df.sort_values(by="id", inplace=True)
                video_predictions = [
                    {k: int(v) if k == 'track_id' else v for k, v in m.items() if np.all(pd.notna(v))} for m in
                    video_predictions_df.to_dict(orient="records")]
                with file_path.open("w") as fp:
                    json.dump({"predictions": video_predictions}, fp, indent=2)
                if save_zip:
                    with zipfile.ZipFile(zf_save_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.write(file_path, arcname=f"{save_path.name}/{file_path.name}")

    @staticmethod
    def soccernet_encoding(dataframe: pd.DataFrame, supercategory):
        dataframe["supercategory"] = supercategory
        dataframe = dataframe.replace({np.nan: None})
        if supercategory == "object":
            # Remove detections that don't have mandatory columns
            # Detections with no track_id will therefore be removed and not count as FP at evaluation
            dataframe.dropna(
                subset=[
                    "track_id",
                    "bbox_ltwh",
                    "bbox_pitch",
                ],
                how="any",
                inplace=True,
            )
            dataframe = dataframe.rename(columns={"bbox_ltwh": "bbox_image", "jersey_number": "jersey"})
            dataframe["track_id"] = dataframe["track_id"]
            dataframe["attributes"] = [{"role": x.get("role"), "jersey": x.get("jersey"), "team": x.get("team")} for n, x in dataframe.iterrows()]
            dataframe["id"] = dataframe.index
            dataframe = dataframe[dataframe.columns.intersection(
                ["id", "image_id", "video_id", "track_id", "supercategory",
                 "category_id", "attributes", "bbox_image", "bbox_pitch"])]

            dataframe['bbox_image'] = dataframe['bbox_image'].apply(transform_bbox_image)
        elif supercategory == "camera":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 6
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "01")
            dataframe = dataframe[dataframe.columns.intersection(
                ["id", "image_id", "video_id", "supercategory", "category_id", "parameters",
                 "relative_mean_reproj", "accuracy@5"])
            ]
        elif supercategory == "pitch":
            dataframe["image_id"] = dataframe.index
            dataframe["category_id"] = 5
            dataframe["id"] = dataframe.index.map(lambda x: str(x) + "00")
            dataframe = dataframe[dataframe.columns.intersection(
                ["id", "image_id", "video_id", "supercategory", "category_id", "lines"])]
        dataframe["video_id"] = dataframe["video_id"].apply(str)
        dataframe["image_id"] = dataframe["image_id"].apply(str)
        dataframe["id"] = dataframe["id"].apply(str)
        dataframe = dataframe.map(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        return dataframe


def transform_bbox_image(row):
    row = row.astype(float)
    return {"x": row[0], "y": row[1], "w": row[2], "h": row[3]}


def extract_category(attributes):
    if attributes['role'] == 'goalkeeper':
        team = attributes['team']
        role = "goalkeeper"
        jersey_number = None
        if attributes['jersey'] is not None:
            jersey_number = int(attributes['jersey']) if attributes['jersey'].isdigit() else None
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif attributes['role'] == "player":
        team = attributes['team']
        role = "player"
        jersey_number = None
        if attributes['jersey'] is not None:
            jersey_number = int(attributes['jersey']) if attributes['jersey'].isdigit() is not None else None
        category = f"{role}_{team}_{jersey_number}" if jersey_number is not None else f"{role}_{team}"
    elif attributes['role'] == "referee":
        team = None
        role = "referee"
        jersey_number = None
        # position = additional_info  # TODO no position for referee in json file (referee's position is not specified in the dataset)
        category = f"{role}"
    elif attributes['role'] == "ball":
        team = None
        role = "ball"
        jersey_number = None
        category = f"{role}"
    else:
        assert attributes['role'] == "other"
        team = None
        role = "other"
        jersey_number = None
        category = f"{role}"
    return category
    
    
def dict_to_df_detections(annotation_dict, categories_list):
    df = pd.DataFrame.from_dict(annotation_dict)

    annotations_pitch_camera = df.loc[df['supercategory'] != 'object']   # remove the rows with non-human categories
    
    df = df.loc[df['supercategory'] == 'object']        # remove the rows with non-human categories
    
    df['bbox_ltwh'] = df.apply(lambda row: xywh_to_ltwh([row['bbox_image']['x_center'], row['bbox_image']['y_center'], row['bbox_image']['w'], row['bbox_image']['h']]), axis=1)
    df['team'] = df.apply(lambda row: row['attributes']['team'], axis=1)
    df['team_cluster'] = (df["team"] == "left").astype(float)
    df['role'] = df.apply(lambda row: row['attributes']['role'], axis=1)
    df['jersey_number'] = df.apply(lambda row: row['attributes']['jersey'], axis=1)
    df['position'] = None # df.apply(lambda row: row['attributes']['position'], axis=1)         for now there is no position in the json file
    df['category'] = df.apply(lambda row: extract_category(row['attributes']), axis=1)
    df['track_id'] = df['track_id'].astype(int)
    # df['id'] = df['id']

    columns = ['id', 'image_id', 'track_id', 'bbox_ltwh', 'bbox_pitch', 'team_cluster',
               'team', 'role', 'jersey_number', 'position', 'category']
    df = df[columns]
    
    video_level_categories = list(df['category'].unique())
    
    return df, annotations_pitch_camera, video_level_categories  

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        file_json = json.load(file)
    return file_json

def video_dir_to_dfs(args):
    season_path = args['dataset_path']
    video_folder = args['video_folder']
    league = args['league']
    season = args['season']

    video_folder_path = os.path.join(season_path, video_folder)
    img_folder_path = os.path.join(video_folder_path, 'img1')

    # 获取联赛文件夹路径
    league_path = os.path.dirname(season_path)

    # 读取 sequences_info.json
    sequences_info_path = os.path.join(league_path, 'sequences_info.json')
    if not os.path.exists(sequences_info_path):
        log.warning(f"在 {league_path} 中未找到 'sequences_info.json'。")
        return None

    with open(sequences_info_path, 'r') as f:
        sequences_info = json.load(f)

    # 提取对应联赛的比赛信息
    match_entries = sequences_info.get(league, [])

    # 构建匹配名称
    match_name = f"{season}/{video_folder}"

    # 找到与当前视频文件夹对应的比赛条目
    match_entry = next(
        (entry for entry in match_entries if entry['name'] == match_name),
        None
    )

    if match_entry is None:
        # 未找到，记录警告
        log.warning(f"未找到视频文件夹 {match_name} 的序列信息")
        return None

    video_id = str(match_entry['id'])
    nframes = match_entry['n_frames']

    # 创建 video_metadata
    video_metadata = {
        'id': video_id,
        'name': match_entry['name'],
        'nframes': nframes,
    }

    # 创建 image_metadata
    nframes = int(nframes)
    img_metadata_df = pd.DataFrame({
        'frame': [i for i in range(0, nframes)],
        'id': [f"{video_id}{i:06d}" for i in range(1, nframes + 1)],
        'video_id': video_id,
        'file_path': [os.path.join(img_folder_path, f'{i:06d}.jpg') for i in range(1, nframes + 1)],
    })

    # 没有检测数据，设置为 None
    detections_df = None
    annotations_pitch_camera_df = None
    video_level_categories = []

    return {
        "video_metadata": video_metadata,
        "image_metadata": img_metadata_df,
        "detections": detections_df,
        "annotations_pitch_camera": annotations_pitch_camera_df,
        "video_level_categories": video_level_categories,
    }

def load_set(dataset_path, nvid=-1, vids_filter_set=None):
    video_metadatas_list = []
    image_metadata_list = []
    annotations_pitch_camera_list = []
    detections_list = []
    categories_list = []
    league_name = os.path.basename(dataset_path)  # e.g., 'spain_laliga'
    season_list = os.listdir(dataset_path)
    season_list.sort()

    for season in season_list:
        season_path = os.path.join(dataset_path, season)
        if not os.path.isdir(season_path):
            continue

        match_list = os.listdir(season_path)
        match_list.sort()
        match_list = [m for m in match_list if os.path.isdir(os.path.join(season_path, m))]

        # 应用 'vids_filter_set'（如果提供）
        if vids_filter_set is not None and len(vids_filter_set) > 0:
            match_list = [m for m in match_list if m in vids_filter_set]

        if nvid > 0:
            match_list = match_list[:nvid]

        if len(match_list) == 0:
            log.warning(f"在 '{league_name}' 的 '{season}' 赛季中，应用过滤后没有剩余的比赛。")
            continue  # 跳过没有比赛的赛季

        args_list = []
        for match_folder in match_list:
            args = {
                'dataset_path': season_path,
                'video_folder': match_folder,
                'league': league_name,
                'season': season,
            }
            args_list.append(args)

        pool = Pool()
        for result in progress(pool.imap_unordered(video_dir_to_dfs, args_list), total=len(args_list), desc=f"加载 '{league_name}/{season}' 赛季的比赛"):
            if result is not None:
                video_metadatas_list.append(result["video_metadata"])
                image_metadata_list.append(result["image_metadata"])
                detections_list.append(result["detections"])
                annotations_pitch_camera_list.append(result["annotations_pitch_camera"])
                categories_list += result["video_level_categories"]

    if len(video_metadatas_list) == 0:
        raise ValueError(f"在联赛 '{league_name}' 中未找到任何比赛。请检查您的数据集路径和结构。")

    video_metadata = pd.DataFrame(video_metadatas_list)
    image_metadata = pd.concat(image_metadata_list, ignore_index=True)

    # 过滤 None 值
    detections_list = [d for d in detections_list if d is not None]
    annotations_pitch_camera_list = [a for a in annotations_pitch_camera_list if a is not None]

    if len(detections_list) > 0:
        detections = pd.concat(detections_list, ignore_index=True)
    else:
        detections = None

    if len(annotations_pitch_camera_list) > 0:
        annotations_pitch_camera = pd.concat(annotations_pitch_camera_list, ignore_index=True)
    else:
        annotations_pitch_camera = None

    image_metadata.set_index("id", drop=False, inplace=True)
    image_gt = image_metadata.copy()
    video_metadata.set_index("id", drop=False, inplace=True)

    return TrackingSet(
        video_metadata,
        image_metadata,
        detections,
        image_gt,
    )

def download_dataset(dataset_path, splits=("train", "valid", "test")):
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=str(dataset_path))
    download = Confirm.ask("Do you want to download the "
                           "datasets automatically ? [i]"
                           f"({'/'.join(splits)})[/i]")
    if download:
        mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
                                               split=splits)
        for split in splits:
            log.info(f"Unzipping {split} split...")
            with zipfile.ZipFile(dataset_path/"gamestate-2024"/f"{split}.zip", 'r') as zf:
                zf.extractall(dataset_path / split)
