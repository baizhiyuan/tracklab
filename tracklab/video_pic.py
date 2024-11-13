import os
import subprocess
import multiprocessing

# 源和目标目录
src_dir = "/garage#2/projects/data/videos"
dst_dir = "/garage#2/projects/data/video_fps_5"

def process_video(mkv_path, img1_dir):
    try:
        # 使用 ffmpeg 将视频切片为图片
        output_pattern = os.path.join(img1_dir, "%06d.jpg")
        command = [
            "ffmpeg",
            "-i", mkv_path,
            "-vf", "fps=1,format=yuv420p",
            output_pattern
        ]
        subprocess.run(command, check=True)
        print(f"Processed {mkv_path} and saved images to {img1_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {mkv_path}: {e}")

def process_files():
    try:
        # 获取所有视频文件
        tasks = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".mkv"):
                    mkv_path = os.path.join(root, file)
                    
                    # 构建目标路径并创建目录结构
                    relative_path = os.path.relpath(root, src_dir)
                    img1_dir = os.path.join(dst_dir, relative_path, "img1")
                    os.makedirs(img1_dir, exist_ok=True)
                    
                    # 将每个任务添加到任务列表
                    tasks.append((mkv_path, img1_dir))
        
        # 使用多进程处理视频切片任务
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(process_video, tasks)

    except KeyboardInterrupt:
        print("\nProcess was interrupted by the user. Exiting gracefully.")

if __name__ == "__main__":
    process_files()
# import os
# import subprocess

# # 源和目标目录
# src_dir = "/garage#2/projects/data/videos"
# dst_dir = "/garage#2/projects/data/video_fps_1"

# try:
#     # 遍历源目录中的所有mkv文件
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             if file.endswith(".mkv"):
#                 # 构建文件的完整路径
#                 mkv_path = os.path.join(root, file)
                
#                 # 构建目标路径并创建目录结构
#                 relative_path = os.path.relpath(root, src_dir)
#                 img1_dir = os.path.join(dst_dir, relative_path, "img1")
#                 os.makedirs(img1_dir, exist_ok=True)
                
#                 # 使用ffmpeg将视频切片为图片
#                 output_pattern = os.path.join(img1_dir, "%06d.jpg")
#                 command = [
#                     "ffmpeg",
#                     "-i", mkv_path,
#                     "-vf", "fps=1,format=yuv420p",
#                     output_pattern
#                 ]
#                 subprocess.run(command, check=True)
                
#                 print(f"Processed {mkv_path} and saved images to {img1_dir}")

# except KeyboardInterrupt:
#     print("\nProcess was interrupted by the user. Exiting gracefully.")