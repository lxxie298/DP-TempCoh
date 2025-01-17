import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)
from torchvision import transforms
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask, brush_stroke_mask_for_vid
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data import degradations
from torch.utils.data import Dataset
from decord import VideoReader
from copy import deepcopy
import glob, os


@DATASET_REGISTRY.register()
class FFHQBlindDataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQBlindDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1.4)
            self.nose_enlarge_ratio = opt.get('nose_enlarge_ratio', 1.1)
            self.mouth_enlarge_ratio = opt.get('mouth_enlarge_ratio', 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False  

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        # inpainting mask
        self.gen_inpaint_mask = opt.get('gen_inpaint_mask', False)
        if self.gen_inpaint_mask:
            logger.info(f'generate mask ...')
            # self.mask_max_angle = opt.get('mask_max_angle', 10)
            # self.mask_max_len = opt.get('mask_max_len', 150)
            # self.mask_max_width = opt.get('mask_max_width', 50)
            # self.mask_draw_times = opt.get('mask_draw_times', 10)
            # # print
            # logger.info(f'mask_max_angle: {self.mask_max_angle}')
            # logger.info(f'mask_max_len: {self.mask_max_len}')
            # logger.info(f'mask_max_width: {self.mask_max_width}')
            # logger.info(f'mask_draw_times: {self.mask_draw_times}')

        # perform corrupt
        self.use_corrupt = opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = opt.get('motion_kernel_path', 'basicsr/data/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt and not self.gen_inpaint_mask:
            # degradation configurations
            self.blur_kernel_size = opt['blur_kernel_size']
            self.blur_sigma = opt['blur_sigma']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.downsample_range = opt['downsample_range']
            self.noise_range = opt['noise_range']
            self.jpeg_range = opt['jpeg_range']
            # print
            logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.gt_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.gt_size - components_bbox['right_eye'][0]
            components_bbox['nose'][0] = self.gt_size - components_bbox['nose'][0]
            components_bbox['mouth'][0] = self.gt_size - components_bbox['mouth'][0]
        
        locations_gt = {}
        locations_in = {}
        for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            elif part == 'nose':
                half_len *= self.nose_enlarge_ratio
            elif part == 'mouth':
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc/(self.gt_size//self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        name = osp.basename(gt_path)[:-4]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (512,512))
        
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)

        if self.load_latent_gt:
            if status[0]:
                latent_gt = self.latent_gt_dict['hflip'][name]
            else:
                latent_gt = self.latent_gt_dict['orig'][name]

        if self.crop_components:
            locations_gt, locations_in = self.get_component_locations(name, status)

        # generate in image
        img_in = img_gt
        if self.use_corrupt and not self.gen_inpaint_mask:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in = cv2.filter2D(img_in,-1,k)
            
            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            img_in = cv2.filter2D(img_in, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_in = cv2.resize(img_in, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                img_in = img_in + noise
                img_in = np.clip(img_in, 0, 1)

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.randint(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
                _, encimg = cv2.imencode('.jpg', img_in * 255., encode_param)
                img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.

            # resize to in_size
            img_in = cv2.resize(img_in, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        # if self.gen_inpaint_mask:
        #     inpaint_mask = random_ff_mask(shape=(self.gt_size,self.gt_size), 
        #         max_angle = self.mask_max_angle, max_len = self.mask_max_len, 
        #         max_width = self.mask_max_width, times = self.mask_draw_times)
        #     img_in = img_in * (1 - inpaint_mask.reshape(self.gt_size,self.gt_size,1)) + \
        #              1.0 * inpaint_mask.reshape(self.gt_size,self.gt_size,1)

        #     inpaint_mask = torch.from_numpy(inpaint_mask).view(1,self.gt_size,self.gt_size)

        if self.gen_inpaint_mask:
            img_in = (img_in*255).astype('uint8')
            img_in = brush_stroke_mask(Image.fromarray(img_in))
            img_in = np.array(img_in) / 255.

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_in = self.color_jitter(img_in, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_in = np.tile(img_in[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)

        # round and clip
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(img_in, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)

        return_dict = {'lq': img_in, 'gt': img_gt, 'gt_path': gt_path}

        if self.crop_components:
            return_dict['locations_in'] = locations_in
            return_dict['locations_gt'] = locations_gt

        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt

        # if self.gen_inpaint_mask:
        #     return_dict['inpaint_mask'] = inpaint_mask

        return return_dict


    def __len__(self):
        return len(self.paths)
    

class GFPGAN_degradation(object):
    def __init__(self, 
                blur_sigma = [0.1, 10], 
                blur_kernel_size = 41, 
                downsample_range = [1, 16], 
                noise_range = [0, 20], 
                jpeg_range = [60, 100],
                gray_prob = 0.0,
                color_jitter_prob = 0.0,
                color_jitter_pt_prob = 0.0,
                shift = 20/255.,
                kernel_list = ['iso', 'aniso'],
                kernel_prob = [0.5, 0.5]):
            self.kernel_list = kernel_list
            self.kernel_prob = kernel_prob
            self.blur_kernel_size = blur_kernel_size
            self.blur_sigma = blur_sigma
            self.downsample_range = downsample_range
            self.noise_range = noise_range
            self.jpeg_range = jpeg_range
            self.gray_prob = gray_prob
            self.color_jitter_prob = color_jitter_prob
            self.color_jitter_pt_prob = color_jitter_pt_prob
            self.shift = shift
    
    def degrade_process(self, img_gt):
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)

        h, w = img_gt.shape[:2]
       
        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        self.blur_sigma[0] = np.random.uniform(self.blur_sigma[0],self.blur_sigma[1])
        self.blur_sigma[1] = self.blur_sigma[0] + 0.5
        kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq
    

    def video_degrade_process(self, frames_gt):
        frames_gt = list(frames_gt)
        if random.random() > 0.5:
            frames_gt = [cv2.flip(frame,1) for frame in frames_gt]

        h, w = frames_gt[0].shape[:2]
       
        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            frames_gt = [np.clip(frame + jitter_val, 0, 1) for frame in frames_gt]

        # random grayscale
        if np.random.uniform() < self.gray_prob:
            frames_gt = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_gt]
            frames_gt = [np.tile(frame[:, :, None], [1, 1, 3]) for frame in frames_gt]
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        self.blur_sigma[0] = np.random.uniform(self.blur_sigma[0],self.blur_sigma[1])
        self.blur_sigma[1] = self.blur_sigma[0] + 0.5
        kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
        frames_lq = [cv2.filter2D(frame, -1, kernel) for frame in frames_gt]

        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        frames_lq = [cv2.resize(frame, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR) for frame in frames_lq]
        
        # noise
        if self.noise_range is not None:
            self.noise_range[0] = np.random.uniform(self.noise_range[0],self.noise_range[1])
            self.noise_range[1] = self.noise_range[0]+1
            frames_lq = [degradations.random_add_gaussian_noise(frame, self.noise_range) for frame in frames_lq]
            # img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            self.jpeg_range[0] = np.random.randint(self.jpeg_range[0],self.jpeg_range[1])
            self.jpeg_range[1] = self.jpeg_range[0] + 1
            frames_lq = [degradations.random_add_jpg_compression(frame, self.jpeg_range) for frame in frames_lq]
            # img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        frames_lq = [(np.clip((frame * 255.0).round(), 0, 255) / 255.) for frame in frames_lq]
        # img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        frames_lq = [cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR) for frame in frames_lq]
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return frames_gt, frames_lq
    
@DATASET_REGISTRY.register()
class BVFRDataset(Dataset):
    def __init__(self, opt):
        self.resolution = opt['resolution']

        self.HQ_imgs = glob.glob(os.path.join(opt['path'],'*'))
        tmp = []
        for p in self.HQ_imgs:
            frames_path = glob.glob(os.path.join(p, '*[.jpg|.jpeg|.png]'))
            if len(frames_path) > (opt['frames_size'] * opt['stride']):
                tmp.append(p)
        self.HQ_imgs = tmp

        self.length = len(self.HQ_imgs)
        self.frames_size = opt['frames_size']
        self.stride = opt['stride']

        self.degrader = GFPGAN_degradation()
        # self.txt_path = list(glob.glob("/DATA/datasets/FFHQwithCaptions/caption/*.txt"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frames_path = glob.glob(os.path.join(self.HQ_imgs[index], '*[.jpg|.jpeg|.png]'))
        frames_path = sorted(frames_path, key=lambda x: os.path.basename(x))
        
        all_idx = range(0,len(frames_path),self.stride)
        s = random.randint(0, len(all_idx)-self.frames_size)
        selected_idx = all_idx[s:s+self.frames_size]
        frames_path = [frames_path[id] for id in selected_idx]

        frames = [cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in frames_path]
        frames = [cv2.resize(f, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA) for f in frames]
        frames = np.stack(frames,axis=0).astype(np.float32)
        frames = frames/255.

        frames_gt, frames_lq = self.degrader.video_degrade_process(frames)
        frames_gt = np.stack(frames_gt, axis=0)
        frames_lq = np.stack(frames_lq, axis=0)

        frames_gt = (frames_gt - 0.5) / 0.5
        frames_lq = (frames_lq - 0.5) / 0.5
        frames_gt = torch.from_numpy(frames_gt).permute(0,3,1,2).contiguous() # [c,f,h,w]
        frames_lq = torch.from_numpy(frames_lq).permute(0,3,1,2).contiguous()
        

        return_dict = {'in': frames_lq, 'gt': frames_gt, 'gt_path': self.HQ_imgs[index]}

        return return_dict



@DATASET_REGISTRY.register()
class VideoImageBFRDataset(Dataset):
    def __init__(self, opt):
        
        self.resolution = opt['resolution']

        self.HQ_videos = glob.glob(os.path.join(opt['video_dir'],'*'))
        paths = []
        for p in self.HQ_videos:
            frames_path = glob.glob(os.path.join(p,'*[.jpg|.jpeg|.png]'))
            if len(list(frames_path)) > 1:
                paths.append(p)
        self.HQ_videos = paths
                
        self.length = len(self.HQ_videos)
        self.image_prob = opt['image_prob']

        if opt.get('image_dir', None):
            self.HQ_images = glob.glob(os.path.join(opt['image_dir'],'*[.jpg|.jpeg|.png]'))
        else:
            self.HQ_images = None

        self.degrader = GFPGAN_degradation()
        print("Used video data", len(self.HQ_videos))
        if self.HQ_images is not None:
            print("Used image data", len(self.HQ_images))
        # self.txt_path = list(glob.glob("/DATA/datasets/FFHQwithCaptions/caption/*.txt"))

    def __len__(self):
        return self.length


    def __getitem__(self, index):
        use_image = False
        if self.HQ_images is not None:
            use_image = (random.random() < self.image_prob)

        if use_image:
            index = random.choice(range(len(self.HQ_images)))
            path = self.HQ_images[index]
            image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            image = resize_random_scale_and_random_crop(image, size=self.resolution, scale_range=(0.8,1.0))
        else:
            # frames_path = glob.glob(os.path.join(self.HQ_videos[index], '*[.jpg|.jpeg|.png]'))
            # path = random.choice(frames_path)
            # image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # image = resize_random_scale_and_random_crop(image, size=self.resolution, scale_range=(0.6,0.8))
            while True:
                try:
                    frames_path = glob.glob(os.path.join(self.HQ_videos[index], '*[.jpg|.jpeg|.png]'))
                    path = random.choice(frames_path)
                    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    image = resize_random_scale_and_random_crop(image, size=self.resolution, scale_range=(0.6,1.0))
                    break
                except:
                    print(self.HQ_videos[index])
                    index  = random.choice(range(len(self.HQ_videos)))


        image = image.astype(np.float32)
        image = image/255.
        frames_gt, frames_lq = self.degrader.degrade_process(image)
        frames_gt = np.stack(frames_gt, axis=0)
        frames_lq = np.stack(frames_lq, axis=0)

        frames_gt = (frames_gt - 0.5) / 0.5
        frames_lq = (frames_lq - 0.5) / 0.5
        frames_gt = torch.from_numpy(frames_gt).permute(2,0,1).contiguous() # [c,f,h,w]
        frames_lq = torch.from_numpy(frames_lq).permute(2,0,1).contiguous()

        ret_data = {
            'gt': frames_gt, # [-1,1]
            'in': frames_lq, # [0,1]
            'gt_path': path
        }

        return ret_data


@DATASET_REGISTRY.register()
class HQVideoDataset(data.Dataset):
    def __init__(self, opt):
        self.resolution = opt['resolution']

        self.HQ_videos = glob.glob(os.path.join(opt['path'],'*'))

        self.length = len(self.HQ_videos)
        self.frames_size = opt['frames_size']
        self.stride = opt['stride']

        self.opt = opt
        # self.txt_path = list(glob.glob("/DATA/datasets/FFHQwithCaptions/caption/*.txt"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        reader = VideoReader(self.HQ_videos[index])
        all_idx = range(0,len(reader),self.stride)
        s = random.randint(0, len(all_idx)-self.frames_size)
        selected_idx = all_idx[s:s+self.frames_size]
        frames = reader.get_batch(selected_idx).asnumpy()

        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        if random.random() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
        scale_range = self.opt.get('scale_range', (1.0,1.3))
        frames = resize_random_scale_and_random_crop_video(frames, self.resolution, scale_range=scale_range)
        frames = frames.astype(np.float32)
        frames = frames/255.

        # frames_gt, frames_lq = self.degrader.video_degrade_process(frames)
        # frames_gt = np.stack(frames_gt, axis=0)
        # frames_lq = np.stack(frames_lq, axis=0)

        frames = (frames - 0.5) / 0.5
        frames = torch.from_numpy(frames).permute(0,3,1,2).contiguous() # [f,c,h,w]
        # frames_lq = torch.from_numpy(frames_lq).permute(0,3,1,2).contiguous()
        

        return_dict = {'gt': frames, 'gt_path': self.HQ_videos[index]}

        return return_dict


@DATASET_REGISTRY.register()
class HQVideoDatasetImgSeq(Dataset):
    def __init__(self, opt):
        logger = get_root_logger()
        self.resolution = opt['resolution']

        self.HQ_videos = glob.glob(os.path.join(opt['path'],'*'))
        paths = []
        for p in self.HQ_videos:
            frames_path = glob.glob(os.path.join(p,'*[.jpg|.jpeg|.png]'))
            if len(frames_path) > (opt['frames_size'] * opt['stride']):
                paths.append(p)
        self.HQ_videos = paths
                
        self.length = len(self.HQ_videos)
        self.opt = opt
        degradation_params = opt.get('degradation_params', {})
        self.degrader = GFPGAN_degradation(**degradation_params)
        print("Used video data", len(self.HQ_videos))
        logger.info("Used video data", len(self.HQ_videos))


        images_path = opt.get('image_path', None)
        self.use_image_prob = 0.0
        if images_path is not None:
            self.HQ_images = glob.glob(os.path.join(images_path,'*[.jpg|.jpeg|.png]'))
            self.use_image_prob = opt.get('use_image_prob', 0.0)
            print("Used image data", len(self.HQ_images))
            logger.info("Used image data", len(self.HQ_images))
            logger.info("Used image prob", self.use_image_prob)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if random.random() < self.use_image_prob:
            index = random.choice(range(len(self.HQ_images)))
            path = self.HQ_images[index]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            frames = [image] * self.opt['frames_size']
            frames = np.stack(frames,axis=0).astype(np.float32)

            basename = os.path.basename(path)
        else:
            frames_path = glob.glob(os.path.join(self.HQ_videos[index], '*[.jpg|.jpeg|.png]'))
            frames_path = sorted(frames_path, key=lambda x: os.path.basename(x))
            
            all_idx = range(0,len(frames_path),self.opt['stride'])
            s = random.randint(0, len(all_idx)-self.opt['frames_size'])
            selected_idx = all_idx[s:s+self.opt['frames_size']]
            frames_path = [frames_path[id] for id in selected_idx]

            frames = [cv2.imread(f, cv2.IMREAD_COLOR) for f in frames_path]
            if random.random() < 0.5:
                frames = [cv2.flip(f, 1) for f in frames]
            # frames = [cv2.resize(f, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA) for f in frames]
            frames = align_images_by_width(frames)
            scale_range = self.opt.get('scale_range', [0.6,1.0])
            frames = resize_random_scale_and_random_crop_video(frames, size=self.resolution, scale_range=scale_range)

            basename = self.HQ_videos[index].split('/')[-1]

        frames = frames.astype(np.float32)
        frames = frames/255.

        frames = np.stack(frames, axis=0)
        # frames_lq = np.stack(frames_lq, axis=0)

        frames = (frames - 0.5) / 0.5
        # frames_lq = (frames_lq - 0.5) / 0.5
        frames = torch.from_numpy(frames).permute(0,3,1,2).contiguous() # [c,f,h,w]
        # frames_lq = torch.from_numpy(frames_lq).permute(0,3,1,2).contiguous()
        
        return_dict = {'gt': frames, 'lq': frames, 'gt_path': basename, 'lq_path': basename}

        return return_dict


@DATASET_REGISTRY.register()
class HQVideoDatasetImgSeqV2(data.Dataset):

    def __init__(self, opt):
        super(HQVideoDatasetImgSeqV2, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.video_dir = opt['video_dir']
        self.frames_size = opt['frames_size']
        self.image_dir = opt.get('image_dir', None)
        self.use_image_prob = opt.get('use_image_prob', 0.0)
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1.4)
            self.nose_enlarge_ratio = opt.get('nose_enlarge_ratio', 1.1)
            self.mouth_enlarge_ratio = opt.get('mouth_enlarge_ratio', 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False  

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.video_dir
            if not self.video_dir.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.video_dir, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.video_paths = glob.glob(os.path.join(self.video_dir,'*'))
            tmp = []
            for p in self.video_paths:
                frames_path = glob.glob(os.path.join(p, '*[.jpg|.jpeg|.png]'))
                if len(frames_path) > (opt['frames_size'] * opt['stride']):
                    tmp.append(p)
            self.video_paths = tmp
            self.image_paths = glob.glob(os.path.join(self.image_dir,'*[.jpg|.jpeg|.png]'))


        # inpainting mask
        self.gen_inpaint_mask = opt.get('gen_inpaint_mask', False)
        if self.gen_inpaint_mask:
            logger.info(f'generate mask ...')
            # self.mask_max_angle = opt.get('mask_max_angle', 10)
            # self.mask_max_len = opt.get('mask_max_len', 150)
            # self.mask_max_width = opt.get('mask_max_width', 50)
            # self.mask_draw_times = opt.get('mask_draw_times', 10)
            # # print
            # logger.info(f'mask_max_angle: {self.mask_max_angle}')
            # logger.info(f'mask_max_len: {self.mask_max_len}')
            # logger.info(f'mask_max_width: {self.mask_max_width}')
            # logger.info(f'mask_draw_times: {self.mask_draw_times}')

        # perform corrupt
        self.use_corrupt = opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = opt.get('motion_kernel_path', 'basicsr/data/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt and not self.gen_inpaint_mask:
            # degradation configurations
            self.blur_kernel_size = opt['blur_kernel_size']
            self.blur_sigma = opt['blur_sigma']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.downsample_range = opt['downsample_range']
            self.noise_range = opt['noise_range']
            self.jpeg_range = opt['jpeg_range']
            # print
            logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.gt_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.gt_size - components_bbox['right_eye'][0]
            components_bbox['nose'][0] = self.gt_size - components_bbox['nose'][0]
            components_bbox['mouth'][0] = self.gt_size - components_bbox['mouth'][0]
        
        locations_gt = {}
        locations_in = {}
        for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            elif part == 'nose':
                half_len *= self.nose_enlarge_ratio
            elif part == 'mouth':
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc/(self.gt_size//self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if random.random() < self.use_image_prob:
        # load gt image
            index = random.choice(range(len(self.image_paths)))
            gt_path = self.image_paths[index]
            basename = osp.basename(gt_path)[:-4]
            img_bytes = self.file_client.get(gt_path)
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.resize(img_gt, (512,512))
            frames = [img_gt] * self.frames_size
            latents = None
        else:
            frames_path = glob.glob(os.path.join(self.video_paths[index], '*[.jpg|.jpeg|.png]'))
            frames_path = sorted(frames_path, key=lambda x: os.path.basename(x))
            
            all_idx = range(0,len(frames_path),self.opt['stride'])
            s = random.randint(0, len(all_idx)-self.opt['frames_size'])
            selected_idx = all_idx[s:s+self.opt['frames_size']]
            frames_path = [frames_path[id] for id in selected_idx]

            frames = []
            for f in frames_path:
                img_bytes = self.file_client.get(f)
                img = imfrombytes(img_bytes, float32=True)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                frames.append(img)
            # frames = [cv2.resize(f, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA) for f in frames]
            frames = align_images_by_width(frames)
            scale_range = self.opt.get('scale_range', [0.6,1.0])
            frames = resize_random_scale_and_random_crop_video(frames, size=self.gt_size, scale_range=scale_range)
            frames = list(frames)

            basename = self.video_paths[index].split('/')[-1]
            
        # random horizontal flip
        if random.random() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
            # latents flip

        # generate in image
        frames_in = frames
        frames_gt = deepcopy(frames)
        if self.use_corrupt and not self.gen_inpaint_mask:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                # img_in = cv2.filter2D(img_in,-1,k)
                frames_in = [cv2.filter2D(f,-1,k) for f in frames_in]
            
            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            # img_in = cv2.filter2D(img_in, -1, kernel)
            frames_in = [cv2.filter2D(f, -1, kernel) for f in frames_in]

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            # img_in = cv2.resize(img_in, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR)
            frames_in = [cv2.resize(f, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR) for f in frames_in]

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                frames_in_ = []
                for f in frames_in:
                    noise = np.float32(np.random.randn(*(f.shape))) * noise_sigma
                    f = f + noise
                    f = np.clip(f, 0, 1)
                    frames_in_.append(f)
                frames_in = frames_in_

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.randint(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
                frames_in_ = []
                for f in frames_in:
                    _, encimg = cv2.imencode('.jpg', f * 255., encode_param)
                    f = np.float32(cv2.imdecode(encimg, 1)) / 255.
                    frames_in_.append(f)
                frames_in = frames_in_

            # resize to in_size
            # img_in = cv2.resize(img_in, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)
            frames_in = [cv2.resize(f, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR) for f in frames_in]

        # if self.gen_inpaint_mask:
        #     inpaint_mask = random_ff_mask(shape=(self.gt_size,self.gt_size), 
        #         max_angle = self.mask_max_angle, max_len = self.mask_max_len, 
        #         max_width = self.mask_max_width, times = self.mask_draw_times)
        #     img_in = img_in * (1 - inpaint_mask.reshape(self.gt_size,self.gt_size,1)) + \
        #              1.0 * inpaint_mask.reshape(self.gt_size,self.gt_size,1)

        #     inpaint_mask = torch.from_numpy(inpaint_mask).view(1,self.gt_size,self.gt_size)

        if self.gen_inpaint_mask:
            img_in = (img_in*255).astype('uint8')
            img_in = brush_stroke_mask(Image.fromarray(img_in))
            img_in = np.array(img_in) / 255.

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            # img_in = self.color_jitter(img_in, self.color_jitter_shift)
            frames_in = [self.color_jitter(f, self.color_jitter_shift) for f in frames_in]
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            frames_in_ = []
            for f in frames_in:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = np.tile(f[:, :, None], [1, 1, 3])
                frames_in_.append(f)
            frames_in = frames_in_
            # img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            # img_in = np.tile(img_in[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)
        frames_in = img2tensor(frames_in, bgr2rgb=True, float32=True) # list
        frames_gt = img2tensor(frames_gt, bgr2rgb=True, float32=True) # list

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            # img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)
            frames_in = [self.color_jitter_pt(f, brightness, contrast, saturation, hue) for f in frames_in]

        frames_in = torch.stack(frames_in, dim=0)
        frames_gt = torch.stack(frames_gt, dim=0)
        # round and clip
        # img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.
        frames_in = np.clip((frames_in * 255.0).round(), 0, 255) / 255.
        frames_gt = np.clip((frames_gt * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(frames_in, self.mean, self.std, inplace=True)
        normalize(frames_gt, self.mean, self.std, inplace=True)

        return_dict = {'lq': frames_in, 'gt': frames_gt, 'gt_path': basename, 'lq_path': basename}


        if self.load_latent_gt:
            return_dict['latent_gt'] = latents

        # if self.gen_inpaint_mask:
        #     return_dict['inpaint_mask'] = inpaint_mask

        return return_dict


    def __len__(self):
        return len(self.video_paths)
    
    
@DATASET_REGISTRY.register()
class HQVideoInpainting(data.Dataset):

    def __init__(self, opt):
        super(HQVideoInpainting, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.video_dir = opt['video_dir']
        self.frames_size = opt['frames_size']
        self.image_dir = opt.get('image_dir', None)
        self.use_image_prob = opt.get('use_image_prob', 0.0)
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1.4)
            self.nose_enlarge_ratio = opt.get('nose_enlarge_ratio', 1.1)
            self.mouth_enlarge_ratio = opt.get('mouth_enlarge_ratio', 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False  

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.video_dir
            if not self.video_dir.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.video_dir, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.video_paths = glob.glob(os.path.join(self.video_dir,'*'))
            tmp = []
            for p in self.video_paths:
                frames_path = glob.glob(os.path.join(p, '*[.jpg|.jpeg|.png]'))
                if len(frames_path) > (opt['frames_size'] * opt['stride']):
                    tmp.append(p)
            self.video_paths = tmp
            if self.image_dir is not None:
                self.image_paths = glob.glob(os.path.join(self.image_dir,'*[.jpg|.jpeg|.png]'))


        # inpainting mask
        self.gen_inpaint_mask = opt.get('gen_inpaint_mask', False)
        if self.gen_inpaint_mask:
            logger.info(f'generate mask ...')
            self.mask_max_angle = opt.get('mask_max_angle', 10)
            self.mask_max_len = opt.get('mask_max_len', 150)
            self.mask_max_width = opt.get('mask_max_width', 50)
            self.mask_draw_times = opt.get('mask_draw_times', 10)
            # print
            logger.info(f'mask_max_angle: {self.mask_max_angle}')
            logger.info(f'mask_max_len: {self.mask_max_len}')
            logger.info(f'mask_max_width: {self.mask_max_width}')
            logger.info(f'mask_draw_times: {self.mask_draw_times}')



    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        if random.random() < self.use_image_prob:
        # load gt image
            index = random.choice(range(len(self.image_paths)))
            gt_path = self.image_paths[index]
            basename = osp.basename(gt_path)[:-4]
            img_bytes = self.file_client.get(gt_path)
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gt = cv2.resize(img_gt, (512,512))
            frames = [img_gt] * self.frames_size
            latents = None
        else:
            frames_path = glob.glob(os.path.join(self.video_paths[index], '*[.jpg|.jpeg|.png]'))
            frames_path = sorted(frames_path, key=lambda x: os.path.basename(x))
            
            all_idx = range(0,len(frames_path),self.opt['stride'])
            s = random.randint(0, len(all_idx)-self.opt['frames_size'])
            selected_idx = all_idx[s:s+self.opt['frames_size']]
            frames_path = [frames_path[id] for id in selected_idx]

            frames = []
            for f in frames_path:
                img_bytes = self.file_client.get(f)
                img = imfrombytes(img_bytes, float32=True)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                frames.append(img)
            # frames = [cv2.resize(f, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA) for f in frames]
            frames = align_images_by_width(frames)
            scale_range = self.opt.get('scale_range', [0.6,1.0])
            frames = resize_random_scale_and_random_crop_video(frames, size=self.gt_size, scale_range=scale_range)
            frames = list(frames)

            basename = self.video_paths[index].split('/')[-1]
            
        # random horizontal flip
        if random.random() < 0.5:
            frames = [cv2.flip(f, 1) for f in frames]
            # latents flip

        # generate in image
        frames_in = frames
        frames_gt = deepcopy(frames)
 
        # if self.gen_inpaint_mask:
        #     inpaint_mask = random_ff_mask(shape=(self.gt_size,self.gt_size), 
        #         max_angle = self.mask_max_angle, max_len = self.mask_max_len, 
        #         max_width = self.mask_max_width, times = self.mask_draw_times)
        #     img_in = img_in * (1 - inpaint_mask.reshape(self.gt_size,self.gt_size,1)) + \
        #              1.0 * inpaint_mask.reshape(self.gt_size,self.gt_size,1)

        #     inpaint_mask = torch.from_numpy(inpaint_mask).view(1,self.gt_size,self.gt_size)
        # frames_in = np.stack(frames_in,axis=0)
        frames_in = [Image.fromarray((f*255).astype('uint8')) for f in frames_in]
        frames_in = brush_stroke_mask_for_vid(frames_in)
        frames_in = [np.array(f) / 255. for f in frames_in]
        # frames_in = frames_in / 255.
        # frames_in = [f for f in frames_in]

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)
        frames_in = img2tensor(frames_in, bgr2rgb=True, float32=True) # list
        frames_gt = img2tensor(frames_gt, bgr2rgb=True, float32=True) # list

        frames_in = torch.stack(frames_in, dim=0)
        frames_gt = torch.stack(frames_gt, dim=0)
        # round and clip
        # img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.
        frames_in = np.clip((frames_in * 255.0).round(), 0, 255) / 255.
        frames_gt = np.clip((frames_gt * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(frames_in, self.mean, self.std, inplace=True)
        normalize(frames_gt, self.mean, self.std, inplace=True)

        return_dict = {'lq': frames_in, 'gt': frames_gt, 'gt_path': basename, 'lq_path': basename}


        return return_dict


    def __len__(self):
        return len(self.video_paths)



def resize_random_scale_and_random_crop(img, size, scale_range=(0.8, 1.2)):
    height, width, _ = img.shape

    # 随机选择一个缩放因子
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])

    # 计算新的尺寸，保持宽高比
    new_width = int(width * scale_factor)
    new_width = max(new_width, size)
    new_height = int(height * scale_factor)
    new_height = max(new_height, size)

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 随机裁剪size x size
    x = np.random.randint(0, new_width - size + 1)
    y = np.random.randint(0, new_height - size + 1)
    cropped_img = resized_img[y:y + size, x:x + size]

    return cropped_img


def resize_random_scale_and_random_crop_video(frames, size, scale_range=(0.8, 1.2)):
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    frames_num, height, width, _ = frames.shape

    # 随机选择一个缩放因子
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])

    # 计算新的尺寸，保持宽高比
    new_width = int(width * scale_factor)
    new_width = max(new_width, size)
    new_height = int(height * scale_factor)
    new_height = max(new_height, size)

    # 调整图像大小
    frames = [cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR) for img in frames]
    frames = np.stack(frames, axis=0)

    # 随机裁剪size x size
    x = np.random.randint(0, new_width - size + 1)
    y = np.random.randint(0, new_height - size + 1)
    cropped_frames = frames[:, y:y + size, x:x + size]

    return cropped_frames

def align_images_by_width(img_list):
    if not isinstance(img_list, list):
        img_list = [img_list]
    target_width = 0
    for img in img_list:
        h, w = img.shape[:2]
        target_width = max(target_width, w)
    resized_images = []
    for img in img_list:
        # 获取原始尺寸
        h, w = img.shape[:2]
        
        # 计算缩放比例
        ratio = target_width / w
        
        # 计算新高度
        new_height = int(h * ratio)
        
        # 缩放图像
        resized_img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 添加到结果列表
        resized_images.append(resized_img)
    
    resized_img = np.stack(resized_img,axis=0)
    return resized_images
