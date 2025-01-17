from basicsr.utils.video_util import VideoWriter
import cv2, os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_dir', type=str, )
parser.add_argument('-o', '--output_dir', type=str, )
args = parser.parse_args()

path = glob(os.path.join(args.image_dir, '*.[jp][pn]g'))
frames = []
for img_path in sorted(path):
    img = cv2.imread(img_path)
    frames.append(img)

basename = args.image_dir.split('/')[-1]
os.makedirs(args.output_dir,exist_ok=True)

vidwriter = VideoWriter(os.path.join(args.output_dir, f"{basename}.mp4"), frames[0].shape[1], frames[0].shape[0], 8, None)
for f in frames:
    vidwriter.write_frame(f)
vidwriter.close()