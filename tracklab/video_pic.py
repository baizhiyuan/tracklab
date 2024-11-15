import os
import cv2
import multiprocessing
import shutil

# 源目录和目标目录
src_dir = "/garage#2/projects/data/videos"
dst_dir = "/garage#2/projects/data/video_fps_1_new"

def process_video(mkv_path, img1_dir):
    try:
        # 在处理之前清空 img1_dir 目录中的现有图像文件
        for f in os.listdir(img1_dir):
            file_path = os.path.join(img1_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        cap = cv2.VideoCapture(mkv_path)
        if not cap.isOpened():
            print(f"无法打开视频文件 {mkv_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 1  # 如果 FPS 不可用或无效，使用默认值 1
        
        frame_save_interval = int(round(fps))  # 跳过的帧数，以大约每秒 1 帧进行保存
        frame_count = 0
        success, frame = cap.read()
        
        while success:
            # 每隔 frame_save_interval 帧保存一帧图像（约每秒 1 帧）
            if frame_count % frame_save_interval == 0:
                # 从 000001.jpg 开始为每张图像命名
                output_filename = os.path.join(img1_dir, f"{(frame_count // frame_save_interval) + 1:06d}.jpg")
                cv2.imwrite(output_filename, frame)
            frame_count += 1
            success, frame = cap.read()
        
        cap.release()
        print(f"已处理 {mkv_path} 并将图像保存到 {img1_dir}")
    except Exception as e:
        print(f"处理 {mkv_path} 时发生错误：{e}")

def process_files():
    try:
        tasks = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file in ["1_224p.mkv", "2_224p.mkv"]:
                    mkv_path = os.path.join(root, file)
                    
                    # 构建相对路径，并添加额外层级以区分不同视频
                    relative_path = os.path.relpath(root, src_dir)
                    video_subdir = os.path.splitext(file)[0]  # 对于 '1_224p' 或 '2_224p'
                    img1_dir = os.path.join(dst_dir, relative_path, video_subdir, "img1")
                    os.makedirs(img1_dir, exist_ok=True)
                    
                    # 将任务添加到任务列表中
                    tasks.append((mkv_path, img1_dir))
        
        # 使用多进程，在保留一些 CPU 核心的情况下避免过载
        cpu_count = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=cpu_count) as pool:
            pool.starmap(process_video, tasks)

    except KeyboardInterrupt:
        print("\n用户中断了进程。正在优雅地退出。")

if __name__ == "__main__":
    process_files()