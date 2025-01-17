import argparse
import glob
import numpy as np
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.video_util import VideoWriter
from einops import rearrange
from decord import VideoReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_path', type=str, default='datasets/ffhq/ffhq_512')
    parser.add_argument('-o', '--save_root', type=str, default='./results/vqgan_rec')
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='./weights/net_g_v1.pth')
    parser.add_argument('--save_video_fps', type=int, default=None)
    parser.add_argument('--frames_per_iter', type=int, default=8)
    args = parser.parse_args()

    if args.save_root.endswith('/'):  # solve when path ends with /
        args.save_root = args.save_root[:-1]
    dir_name = os.path.abspath(args.save_root)
    os.makedirs(dir_name, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path = args.test_path
    save_root = args.save_root
    ckpt_path = args.ckpt_path
    codebook_size = args.codebook_size

    vqgan = ARCH_REGISTRY.get('VideoCodeFormerStage2p5')(dim_embd=512, n_head=8, n_layers=9,
                codebook_size=1024, latent_size=256, st_latent_size=2048,
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantize','generator'], vqgan_path=None).to(device)
    checkpoint = torch.load(ckpt_path)['params_ema']
    vqgan.load_state_dict(checkpoint,strict=True)
    vqgan.eval()

    for video_path in sorted(glob.glob(os.path.join(test_path, '*.mp4'))):
        video_name = os.path.basename(video_path)
        print(video_name)
        input_img_list = []
        vidreader = VideoReader(video_path)
        fps = vidreader.get_avg_fps()
        input_img_list = vidreader.get_batch(range(0,len(vidreader))).asnumpy()
        input_img_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in input_img_list]

        input_img_list = [cv2.resize(f,(512,512)) for f in input_img_list]
        frames = np.stack(input_img_list, axis=0).astype(np.float32)
        frames = frames / 255.
        frames = torch.from_numpy(frames).permute(0,3,1,2) # [f c h w]
        normalize(frames, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        frames = frames.unsqueeze(0).to(device)
        output = []
        while frames.shape[1] % args.frames_per_iter != 0:
            frames = torch.cat([frames, frames[:,-1:]], dim=1)
        with torch.no_grad():
            for k in range(0, frames.shape[1]-args.frames_per_iter+1, args.frames_per_iter//2):
                frames_clip = frames[:,k:k+args.frames_per_iter]
                o = vqgan(frames_clip)[0] # [f c h w]
                if k>0:
                    output.append(o[:,args.frames_per_iter//2:])
                else:
                    output.append(o)
        output = torch.cat(output, dim=1)

        output = output.clip(-1, 1)
        output = ((output[0]+1)/2) * 255.
        frames = ((frames[0]+1)/2) * 255
        output = output.permute(0,2,3,1)
        frames = frames.permute(0,2,3,1)
        output = output.detach().cpu().numpy().astype('uint8')
        frames = frames.detach().cpu().numpy().astype('uint8')
        output = np.concatenate([frames, output], axis=2)
        torch.cuda.empty_cache()
        path = os.path.join(save_root, video_name)
        vidwriter = VideoWriter(path, height=output.shape[1], width=output.shape[2], fps=fps, audio=None)
        for f in output:
            vidwriter.write_frame(f)
        vidwriter.close()

    print(f'\nAll results are saved in {save_root}')

