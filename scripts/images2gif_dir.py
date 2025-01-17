from PIL import Image
import os
import argparse
import re
from glob import glob

def extract_numbers(s):
    return re.findall(r'\d+', s)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_dir', type=str, )
parser.add_argument('-o', '--output_dir', type=str, )
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# 基础路径，包含多个图片目录
base_path = args.image_dir

# 获取所有图片目录
directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


# 遍历每个目录，创建 GIF
for directory in directories:
    image_dir = os.path.join(base_path, directory)
    file_list = glob(os.path.join(image_dir, '*.[jp][pn]g'))
    images = []
    # 按顺序加载图片
    # file_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    file_list = sorted(file_list, key=lambda x:int(extract_numbers(os.path.basename(x))[-1]))
    for filename in file_list:
        file_path = os.path.join(image_dir, filename)
        images.append(Image.open(file_path))
    
    # 创建 GIF
    try:
        # os.makedirs(os.path.join(args.output_dir, directory),exist_ok=True)
        output_gif_path = os.path.join(args.output_dir,f'{directory}.gif')
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    except:
        pass
    
    print(f"Created GIF at {output_gif_path}")
