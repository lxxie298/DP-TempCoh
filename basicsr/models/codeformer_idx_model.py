import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel
from einops import rearrange
from basicsr.utils.video_util import VideoReader, VideoWriter
import os
import numpy as np

@MODEL_REGISTRY.register()
class CodeFormerIdxModel(SRModel):
    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        # if self.gt.dim() == 5:
        #     self.gt = rearrange(self.gt,'b f c h w -> (b f) c h w')
        if self.gt.dim() == 4:
            self.gt = self.gt.unsqueeze(1)
        self.input = data['lq'].to(self.device)
        # if self.input.dim() == 5:
        #     self.input = rearrange(self.input,'b f c h w -> (b f) c h w')
        if self.input.dim() == 4:
            self.input = self.input.unsqueeze(1)
        self.b = self.gt.shape[0]

        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            self.idx_gt = self.idx_gt.view(self.b, -1)
        else:
            self.idx_gt = None

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if self.opt['datasets']['train'].get('latent_gt_path', None) is not None:
            self.generate_idx_gt = False
        elif self.opt.get('network_vqgan', None) is not None:
            self.hq_vqgan_fix = build_network(self.opt['network_vqgan']).to(self.device)
            self.hq_vqgan_fix.eval()
            self.generate_idx_gt = True
            for param in self.hq_vqgan_fix.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(f'Shoule have network_vqgan config or pre-calculated latent code.')
        
        logger.info(f'Need to generate latent GT code: {self.generate_idx_gt}')

        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', True)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', True)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)

        self.net_g.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()

        if self.generate_idx_gt:
            x = self.hq_vqgan_fix.encoder(self.gt)
            _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']
            self.idx_gt = min_encoding_indices.view(self.b, -1)
        
        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g,'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[self.b,16,16,256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[self.b,16,16,256])

        logits, lq_feat = self.net_g(self.input, w=0, code_only=True)

        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(hw)n -> bn(hw)
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['cross_entropy_loss'] = cross_entropy_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output, _, _ = self.net_g_ema(self.input, w=0)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _, _ = self.net_g(self.input, w=0)
                self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            lq_img = tensor2img([visuals['lq']],min_max=(-1,1))
            sr_img = tensor2img([visuals['result']],min_max=(-1,1))
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']],min_max=(-1,1))
                del self.gt

            # tentative for out of GPU memory
            del self.input
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                
                sr_img = np.concatenate([lq_img, sr_img, gt_img],axis=1)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.input.detach().cpu()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict


    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class VideoCodeFormerIdxModel(CodeFormerIdxModel):
    def init_training_settings(self):
        super().init_training_settings()
        # reweight module
        if hasattr(self.net_g, 'module'):
            self.net_g.module._init_spatialtemporal_position_weight(8)
        else:
            self.net_g._init_spatialtemporal_position_weight(8)
        if hasattr(self.net_g_ema, 'module'):
            self.net_g_ema.module._init_spatialtemporal_position_weight(8)
        else:
            self.net_g_ema._init_spatialtemporal_position_weight(8)

    def feed_data(self, data):
        batch_size, frames_size, _, height, width = data['gt'].shape
        self.gt = data['gt'].to(self.device)
        if self.gt.dim() == 4:
            self.gt = self.gt.unsqueeze(1)
        self.input = data['lq'].to(self.device)
        if self.input.dim() == 4:
            self.input = self.input.unsqueeze(1)
        self.b = self.gt.shape[0]
        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            self.idx_gt = self.idx_gt.view(self.b, -1)
        else:
            self.idx_gt = None
        if frames_size > 8:
            self.gt_pre = self.gt[:,:8]
            self.gt = self.gt[:, 4:]
            self.input_pre = self.input[:,:8]
            self.input = self.input[:, 4:]

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()
        batch_size, frame_size = self.gt.shape[0], self.gt.shape[1]

        if self.generate_idx_gt:
            gt = rearrange(self.gt, 'b f c h w -> (b f) c h w')
            x = self.hq_vqgan_fix.encoder(gt)
            _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']  # (bfhw)
            self.idx_gt = min_encoding_indices.view(batch_size, -1) # b(fhw)
        
        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g,'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256]) # (bf,256,16,16)
            quant_feat_gt = rearrange(quant_feat_gt, '(b f) c h w -> b f c h w', f=frame_size)

        logits, lq_feat, consistency_loss = self.net_g(self.input, w=0, code_only=True,  consistency_loss=True)

        with torch.no_grad():
            logits_pre, _, _ = self.net_g_ema(self.input_pre, w=0, code_only=True, consistency_loss=False)


        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(fhw)n -> bn(fhw)
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['cross_entropy_loss'] = cross_entropy_loss

        if consistency_loss is not None:
            l_g_total += consistency_loss
            loss_dict['consistency_loss'] = consistency_loss

        logits_pre = rearrange(logits_pre, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,frame_size//2:]
        logits = rearrange(logits, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,:frame_size//2]
        ema_loss = F.l1_loss(logits, logits_pre) * 0.5
        l_g_total += ema_loss
        loss_dict['ema_loss'] = ema_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            video_name = val_data['lq_path'][0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']])
            sr_img = visuals['result']
            batch_size, frame_size = sr_img.shape[0], sr_img.shape[1]
            sr_img = sr_img.clip(-1,1)
            sr_img = ((sr_img+1)/2)*255
            sr_img = sr_img.detach().cpu().numpy().astype('uint8')
            sr_img = rearrange(sr_img, 'b f c h w -> b f h w c')
            sr_img = sr_img[0]

            input_img = self.input
            input_img = input_img.clip(-1,1)
            input_img = ((input_img+1)/2)*255
            input_img = input_img.detach().cpu().numpy().astype('uint8')
            input_img = rearrange(input_img, 'b f c h w -> b f h w c')
            input_img = input_img[0]

            if 'gt' in visuals:
                gt_img = visuals['gt']
                gt_img = gt_img.clip(-1,1)
                gt_img = ((gt_img+1)/2)*255
                gt_img = gt_img.detach().cpu().numpy().astype('uint8')
                gt_img = rearrange(gt_img, 'b f c h w -> b f h w c')
                gt_img = gt_img[0]
                del self.gt

            # tentative for out of GPU memory
            del self.input
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], video_name,
                                             f'{video_name}_{current_iter}.mp4')
                    os.makedirs(osp.join(self.opt['path']['visualization'], video_name), exist_ok=True)
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{video_name}_{self.opt["val"]["suffix"]}.mp4')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{video_name}_{self.opt["name"]}.mp4')
                    os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name), exist_ok=True)
                
                if 'gt' in visuals:
                    out_img = np.concatenate([input_img, sr_img, gt_img],axis=2)
                else:
                    out_img = np.concatenate([input_img, sr_img],axis=2)
                
                vidwriter = VideoWriter(save_img_path, out_img.shape[1], out_img.shape[2], 8, None)
                for f in out_img:
                    vidwriter.write_frame(f)
                vidwriter.close()


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {video_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
@MODEL_REGISTRY.register()  
class VideoCodeFormerIdxModelSAware(VideoCodeFormerIdxModel):
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()
        batch_size, frame_size = self.gt.shape[0], self.gt.shape[1]

        if self.generate_idx_gt:
            gt = rearrange(self.gt, 'b f c h w -> (b f) c h w')
            x = self.hq_vqgan_fix.encoder(gt)
            _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']  # (bfhw)
            self.idx_gt = min_encoding_indices.view(batch_size, -1) # b(fhw)
        
        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g,'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256]) # (bf,256,16,16)
            quant_feat_gt = rearrange(quant_feat_gt, '(b f) c h w -> b f c h w', f=frame_size)

        logits, lq_feat, consistency_loss = self.net_g(self.input, w=0, code_only=True,  consistency_loss=True)

        with torch.no_grad():
            logits_pre, _, _ = self.net_g_ema(self.input_pre, w=0, code_only=True, consistency_loss=False)


        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(fhw)n -> bn(fhw)
            self.idx_gt = rearrange(self.idx_gt, 'b (f h w) -> (b f) (h w)', f=frame_size,h=16,w=16)
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['cross_entropy_loss'] = cross_entropy_loss

        if consistency_loss is not None:
            l_g_total += consistency_loss
            loss_dict['consistency_loss'] = consistency_loss

        logits_pre = rearrange(logits_pre, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,frame_size//2:]
        logits = rearrange(logits, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,:frame_size//2]
        ema_loss = F.l1_loss(logits, logits_pre)
        l_g_total += ema_loss
        loss_dict['ema_loss'] = ema_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
            
@MODEL_REGISTRY.register() 
class VideoCodeFormerIdxInpaintingModel(VideoCodeFormerIdxModel):
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()
        batch_size, frame_size = self.gt.shape[0], self.gt.shape[1]
        
        x_ = rearrange(self.input, 'b f c h w -> b (f c) h w')
        all_ones = (x_ == 1).all(dim=1, keepdim=True) # B 1 H W
        mask = all_ones.float()
        mask = mask.repeat(1,frame_size,1,1) # B F H W
        # mask = F.interpolate(mask, (16,16), mode='nearest')
        mask = F.adaptive_max_pool2d(mask, (16,16))
        mask = mask.long()

        if self.generate_idx_gt:
            gt = rearrange(self.gt, 'b f c h w -> (b f) c h w')
            x = self.hq_vqgan_fix.encoder(gt)
            _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']  # (bfhw)
            self.idx_gt = min_encoding_indices.view(batch_size, -1) # b(fhw)
        
        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g,'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256]) # (bf,256,16,16)
            quant_feat_gt = rearrange(quant_feat_gt, '(b f) c h w -> b f c h w', f=frame_size)

        logits, lq_feat, consistency_loss = self.net_g(self.input, w=0, code_only=True,  consistency_loss=True)

        with torch.no_grad():
            logits_pre, _, _ = self.net_g_ema(self.input_pre, w=0, code_only=True, consistency_loss=False)


        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(fhw)n -> bn(fhw)
            mask_logits = rearrange(mask, 'b f h w -> b (f h w)')
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt, reduction='none') * mask_logits * self.entropy_loss_weight
            # print(mask_logits.shape)
            # print(cross_entropy_loss.shape)
            l_g_total += cross_entropy_loss.mean() * self.entropy_loss_weight
            loss_dict['cross_entropy_loss'] = cross_entropy_loss.mean()

        if consistency_loss is not None:
            l_g_total += consistency_loss
            loss_dict['consistency_loss'] = consistency_loss

        logits_pre = rearrange(logits_pre, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,frame_size//2:]
        logits = rearrange(logits, 'b (f h w) n -> b f n h w', f=frame_size, w=lq_feat.shape[-1])[:,:frame_size//2]
        ema_loss = F.l1_loss(logits, logits_pre)
        l_g_total += ema_loss
        loss_dict['ema_loss'] = ema_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
        
        
@MODEL_REGISTRY.register()
class VideoCodeFormerIdxModelV2(CodeFormerIdxModel):
    def init_training_settings(self):
        super().init_training_settings()
        # reweight module
        if hasattr(self.net_g, 'module'):
            self.net_g.module._init_spatialtemporal_position_weight(8)
        else:
            self.net_g._init_spatialtemporal_position_weight(8)
        if hasattr(self.net_g_ema, 'module'):
            self.net_g_ema.module._init_spatialtemporal_position_weight(8)
        else:
            self.net_g_ema._init_spatialtemporal_position_weight(8)

    def feed_data(self, data):
        batch_size, frames_size, _, height, width = data['gt'].shape
        self.gt = data['gt'].to(self.device)
        if self.gt.dim() == 4:
            self.gt = self.gt.unsqueeze(1)
        self.input = data['lq'].to(self.device)
        if self.input.dim() == 4:
            self.input = self.input.unsqueeze(1)
        self.b = self.gt.shape[0]
        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            self.idx_gt = self.idx_gt.view(self.b, -1)
        else:
            self.idx_gt = None

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        self.optimizer_g.zero_grad()
        batch_size, frame_size = self.gt.shape[0], self.gt.shape[1]

        if self.generate_idx_gt:
            gt = rearrange(self.gt, 'b f c h w -> (b f) c h w')
            x = self.hq_vqgan_fix.encoder(gt)
            _, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']  # (bfhw)
            self.idx_gt = min_encoding_indices.view(batch_size, -1) # b(fhw)
        
        if self.hq_feat_loss:
            # quant_feats
            if hasattr(self.net_g,'module'):
                quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256])
            else:
                quant_feat_gt = self.net_g.quantize.get_codebook_feat(self.idx_gt, shape=[int(batch_size*frame_size),16,16,256]) # (bf,256,16,16)
            quant_feat_gt = rearrange(quant_feat_gt, '(b f) c h w -> b f c h w', f=frame_size)

        logits, lq_feat = self.net_g(self.input, w=0, code_only=True)


        l_g_total = 0
        loss_dict = OrderedDict()
        # hq_feat_loss
        if self.hq_feat_loss: # codebook loss 
            l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
            l_g_total += l_feat_encoder
            loss_dict['l_feat_encoder'] = l_feat_encoder

        # cross_entropy_loss
        if self.cross_entropy_loss:
            # b(fhw)n -> bn(fhw)
            cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
            l_g_total += cross_entropy_loss
            loss_dict['cross_entropy_loss'] = cross_entropy_loss

        l_g_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            video_name = val_data['lq_path'][0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            # sr_img = tensor2img([visuals['result']])
            sr_img = visuals['result']
            batch_size, frame_size = sr_img.shape[0], sr_img.shape[1]
            sr_img = sr_img.clip(-1,1)
            sr_img = ((sr_img+1)/2)*255
            sr_img = sr_img.detach().cpu().numpy().astype('uint8')
            sr_img = rearrange(sr_img, 'b f c h w -> b f h w c')
            sr_img = sr_img[0]

            input_img = self.input
            input_img = input_img.clip(-1,1)
            input_img = ((input_img+1)/2)*255
            input_img = input_img.detach().cpu().numpy().astype('uint8')
            input_img = rearrange(input_img, 'b f c h w -> b f h w c')
            input_img = input_img[0]

            if 'gt' in visuals:
                gt_img = visuals['gt']
                gt_img = gt_img.clip(-1,1)
                gt_img = ((gt_img+1)/2)*255
                gt_img = gt_img.detach().cpu().numpy().astype('uint8')
                gt_img = rearrange(gt_img, 'b f c h w -> b f h w c')
                gt_img = gt_img[0]
                del self.gt

            # tentative for out of GPU memory
            del self.input
            del self.output
            torch.cuda.empty_cache()

            if save_img:

                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], video_name,
                                             f'{video_name}_{current_iter}.mp4')
                    os.makedirs(osp.join(self.opt['path']['visualization'], video_name), exist_ok=True)
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{video_name}_{self.opt["val"]["suffix"]}.mp4')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{video_name}_{self.opt["name"]}.mp4')
                    os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name), exist_ok=True)
                
                if 'gt' in visuals:
                    out_img = np.concatenate([input_img, sr_img, gt_img],axis=2)
                else:
                    out_img = np.concatenate([input_img, sr_img],axis=2)
                
                vidwriter = VideoWriter(save_img_path, out_img.shape[1], out_img.shape[2], 8, None)
                for f in out_img:
                    vidwriter.write_frame(f)
                vidwriter.close()


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {video_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)