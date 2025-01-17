from decord import VideoReader
from decord import cpu
import imageio
import os
from tqdm import tqdm
import argparse
import re

def extract_numbers(s):
    return re.findall(r'\d+', s)


parser = argparse.ArgumentParser()
parser.add_argument('-i','--video_dir')
parser.add_argument('-o','--output_dir')
args = parser.parse_args()

# 输入视频文件的目录路径
input_directory = args.video_dir

# 输出图片的根目录路径
output_root = args.output_dir
if not os.path.exists(output_root):
    os.makedirs(output_root)

# 获取目录中所有视频文件
video_files = [f for f in os.listdir(input_directory) if f.endswith(('.mp4', '.avi'))]

# 遍历每个视频文件
for video_file in tqdm(video_files):
    video_path = os.path.join(input_directory, video_file)
    output_folder = os.path.join(output_root, os.path.splitext(video_file)[0])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用decord读取视频
    vr = VideoReader(video_path, ctx=cpu(0))

    # 遍历视频中的每一帧
    for i in range(len(vr)):
        frame = vr[i].asnumpy()  # 将视频帧转换为NumPy数组
        image_path = os.path.join(output_folder, f'frame_{i:04d}.jpg')
        imageio.imwrite(image_path, frame)  # 保存帧为JPEG图片

    print(f"已将视频 {video_file} 的所有帧保存至: {output_folder}")

print("所有视频帧已处理完成.")
