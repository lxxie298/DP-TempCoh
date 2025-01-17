'''
VQGAN code, adapted from the original created by the Unleashing Transformers authors:
https://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.temporal_arch import SpatioTemporalResBlockSon, TransformerSpatioTemporalModelSon
from einops import rearrange
from torch.utils.checkpoint import checkpoint
def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    

@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)


#  Define VQVAE classes
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous() # (bf)chw -> (bf)hwc
        z_flattened = z.view(-1, self.emb_dim) # (bf)hwc -> (bfhw)c

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # (bfhw)1
        _, min_encoding_indices_topk = torch.topk(d, 8, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        # min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance,
            "min_encoding_indices_topk": min_encoding_indices_topk
            }

    def get_codebook_feat(self, indices, shape):
        # input indices: batch*token_num -> (batch*token_num)*1 or (bfhw)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, num_hiddens, straight_through=False, kl_weight=5e-4, temp_init=1.0):
        super().__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, codebook_size, 1)  # projects last encoder layer to quantized logits
        self.embed = nn.Embedding(codebook_size, emb_dim)

    def forward(self, z):
        hard = self.straight_through if self.training else True

        logits = self.proj(z)

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)

        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_size + 1e-10), dim=1).mean()
        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, diff, {
            "min_encoding_indices": min_encoding_indices
        }


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k) 
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1) 
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions, use_grad_checkpoint=True):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,)+tuple(ch_mult)

        blocks = []
        # initial convultion
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))

        # residual and downsampling blocks, with attention on smaller res (16x16)
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # normalise and convert to latent size
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)
        self.use_grad_checkpoint = use_grad_checkpoint

    # def forward(self, x):
    #     for block in self.blocks:
    #         x = block(x)
            
    #     return x
    
    def forward(self, x):
        for block in self.blocks:
            if isinstance(block, (ResBlock, AttnBlock)) and self.use_grad_checkpoint:
                # 当开关打开时，使用 checkpoint
                x = checkpoint(self.run_block, block, x)
            else:
                x = block(x)
        return x

    @staticmethod
    def run_block(block, x):
        # Helper function to execute a block
        return block(x)

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf 
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size 
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch

                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)
   

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        return x
    

class VideoGenerator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions, use_grad_checkpoint=True):
        super().__init__()
        self.nf = nf 
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size 
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        temporal_blocks = []
        temporal_attens = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        temporal_blocks.append(SpatioTemporalResBlockSon(block_in_ch,block_in_ch, temb_channels=None))
        blocks.append(AttnBlock(block_in_ch))
        temporal_attens.append(TransformerSpatioTemporalModelSon(
            in_channels = block_in_ch,
        ))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        temporal_blocks.append(SpatioTemporalResBlockSon(block_in_ch,block_in_ch, temb_channels=None))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                temporal_blocks.append(SpatioTemporalResBlockSon(block_out_ch,block_out_ch, temb_channels=None))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
                    temporal_attens.append(TransformerSpatioTemporalModelSon(
                        in_channels = block_in_ch,
                    ))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)
        self.temporal_blocks = nn.ModuleList(temporal_blocks)
        self.temporal_attens = nn.ModuleList(temporal_attens)
        self.use_grad_checkpoint = use_grad_checkpoint
   

    # def forward(self, x, image_only_indicator=None):
    #     temporal_blocks = self.temporal_blocks
    #     temporal_attens = self.temporal_attens

    #     for block in self.blocks:
    #         x = block(x)
    #         if isinstance(block, ResBlock):
    #             block_ = temporal_blocks[0]
    #             x = block_(x, image_only_indicator=image_only_indicator)
    #             temporal_blocks = temporal_blocks[1:]
    #         elif isinstance(block, AttnBlock):
    #             block_ = temporal_attens[0]
    #             x = block_(x,image_only_indicator=image_only_indicator)
    #             temporal_attens = temporal_attens[1:]
    #     return x
        
    def forward(self, x, image_only_indicator=None):
        temporal_blocks_iter = list(self.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.temporal_attens)[::-1]

        for i, block in enumerate(self.blocks):
            # 当 block 是 ResBlock 或 AttnBlock 时使用 checkpoint
            if isinstance(block, (ResBlock, AttnBlock)):
                if self.use_grad_checkpoint:
                    x = checkpoint(block, x)
                    if isinstance(block, ResBlock):
                        block_ = temporal_blocks_iter.pop()
                        if len(temporal_blocks_iter) > 0:
                            x = checkpoint(block_, x, None, image_only_indicator)
                        else:
                            x = block_(x, image_only_indicator=image_only_indicator)
                    elif isinstance(block, AttnBlock):
                        block_ = temporal_attens_iter.pop()
                        x = checkpoint(block_, x, None, image_only_indicator)
                else:
                    x = block(x)
                    if isinstance(block, ResBlock):
                        block_ = temporal_blocks_iter.pop()
                        x = block_(x, image_only_indicator=image_only_indicator)
                    elif isinstance(block, AttnBlock):
                        block_ = temporal_attens_iter.pop()
                        x = block_(x, image_only_indicator=image_only_indicator)
            else:
                x = block(x)
        return x
    

class VideoGeneratorV2(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions, use_grad_checkpoint=True):
        super().__init__()
        self.nf = nf 
        self.ch_mult = ch_mult 
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size 
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions-1)

        blocks = []
        temporal_blocks = []
        temporal_attens = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))

        # non-local attention block
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        temporal_blocks.append(SpatioTemporalResBlockSon(block_in_ch,block_in_ch, temb_channels=None))
        blocks.append(AttnBlock(block_in_ch))
        temporal_attens.append(TransformerSpatioTemporalModelSon(
            in_channels = block_in_ch,
            attention_head_dim = 16,
        ))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        temporal_blocks.append(SpatioTemporalResBlockSon(block_in_ch,block_in_ch, temb_channels=None))

        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]

            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                temporal_blocks.append(SpatioTemporalResBlockSon(block_out_ch,block_out_ch, temb_channels=None))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
                    temporal_attens.append(TransformerSpatioTemporalModelSon(
                        in_channels = block_in_ch,
                        attention_head_dim = 16,
                    ))

            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))

        self.blocks = nn.ModuleList(blocks)
        self.temporal_blocks = nn.ModuleList(temporal_blocks)
        self.temporal_attens = nn.ModuleList(temporal_attens)
        self.use_grad_checkpoint = use_grad_checkpoint
   

    # def forward(self, x, image_only_indicator=None):
    #     temporal_blocks = self.temporal_blocks
    #     temporal_attens = self.temporal_attens

    #     for block in self.blocks:
    #         x = block(x)
    #         if isinstance(block, ResBlock):
    #             block_ = temporal_blocks[0]
    #             x = block_(x, image_only_indicator=image_only_indicator)
    #             temporal_blocks = temporal_blocks[1:]
    #         elif isinstance(block, AttnBlock):
    #             block_ = temporal_attens[0]
    #             x = block_(x,image_only_indicator=image_only_indicator)
    #             temporal_attens = temporal_attens[1:]
    #     return x
        
    def forward(self, x, image_only_indicator=None):
        temporal_blocks_iter = list(self.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.temporal_attens)[::-1]

        for i, block in enumerate(self.blocks):
            # 当 block 是 ResBlock 或 AttnBlock 时使用 checkpoint
            if isinstance(block, (ResBlock, AttnBlock)):
                if self.use_grad_checkpoint:
                    x = checkpoint(block, x)
                    if isinstance(block, ResBlock):
                        block_ = temporal_blocks_iter.pop()
                        if len(temporal_blocks_iter) > 0:
                            x = checkpoint(block_, x, None, image_only_indicator)
                        else:
                            x = block_(x, image_only_indicator=image_only_indicator)
                    elif isinstance(block, AttnBlock):
                        block_ = temporal_attens_iter.pop()
                        x = checkpoint(block_, x, None, image_only_indicator)
                else:
                    x = block(x)
                    if isinstance(block, ResBlock):
                        block_ = temporal_blocks_iter.pop()
                        x = block_(x, image_only_indicator=image_only_indicator)
                    elif isinstance(block, AttnBlock):
                        block_ = temporal_attens_iter.pop()
                        x = block_(x, image_only_indicator=image_only_indicator)
            else:
                x = block(x)
        return x

  
@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = Generator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'])
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')


    def forward(self, x):
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        x = self.generator(quant)
        return x, codebook_loss, quant_stats


@ARCH_REGISTRY.register()
class TemporalVQVAE(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None, freeze_backbone=True, training_module_names=[], use_grad_checkpoint=True, motion_bank_size=None):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
            if motion_bank_size is not None:
                self.motion_quantize = VectorQuantizer(motion_bank_size, 16, self.beta)
            else:
                self.motion_quantize = VectorQuantizer(self.codebook_size*16, 16, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = VideoGenerator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'], strict=True)
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'],strict=True)
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')
            
        # freeze backbone
        if freeze_backbone:
            for name, p in self.named_parameters():
                p.requires_grad_(False)
                for each in training_module_names:
                    if each in name:
                        p.requires_grad_(True)

    def get_quant_motion(self, x):
        assert x.dim() == 5, 'shape must be (b f c h w)'
        quant_motion_mean = torch.mean(x, dim=2, keepdim=True)
        quant_motion_var = torch.var(x, dim=2, keepdim=True)
        quant_motion = torch.cat([quant_motion_mean, quant_motion_var], dim=2)
        quant_motion = rearrange(quant_motion, 'b f c h w -> b (f c) h w')
        quant_motion, quant_loss, _ = self.motion_quantize(quant_motion)
        return quant_motion, quant_loss

    def modulate_motion(self, x, motion_mean_var, eps=1e-6):
        assert x.dim() == 5, 'shape must be (b f c h w)'
        assert motion_mean_var.dim() == 4, 'shape must be (b (2f) h w)'
        motion_mean = torch.mean(x, dim=2, keepdim=True)
        motion_var = torch.var(x, dim=2, keepdim=True)

        motion_mean_var = rearrange(motion_mean_var, 'b (f c) h w -> b f c h w', c=2)
        new_mean, new_var = motion_mean_var.chunk(2, dim=2)
        x = (x - motion_mean) / torch.sqrt(motion_var + eps)
        x = x * torch.sqrt(new_var) + new_mean

        return x


    def forward(self, x):
        batch_size, frame_size, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        quant_motion = rearrange(quant, '(b f) c h w -> b f c h w', f=frame_size)
        quant_motion, quant_motion_loss = self.get_quant_motion(quant_motion)
        quant = rearrange(quant, '(b f) c h w -> b f c h w', f=frame_size)
        quant = self.modulate_motion(quant, quant_motion)
        quant = rearrange(quant, 'b f c h w -> (b f) c h w', f=frame_size)

        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = self.generator(quant, image_only_indicator=image_only_indicator)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        return x, (codebook_loss, quant_motion_loss), quant_stats

    
    def forward_dycnamic_dict(self, x, frame_size=8):
        batch_size, total_frames, _, _, _ = x.shape
        latents = []
        global_latent = None
        for k in range(0, total_frames, frame_size):
            frame_clip = x[:,k:k+frame_size]
            frame_clip = rearrange(frame_clip, 'b f c h w -> (b f) c h w')
            print(frame_clip.shape)
            frame_clip = self.encoder(frame_clip)
            print(frame_clip.shape)
            quant, _, _ = self.quantize(frame_clip)

            if global_latent is None:
                quant = rearrange(quant, '(b f) c h w -> b f c h w', f=frame_size)
                latents.append(quant)
                global_latent = rearrange(quant, 'b f c h w -> b (f h w) c')
            else:
                height, width = quant.shape[-2], quant.shape[-1]
                quant = rearrange(quant, '(b f) c h w -> b (f h w) c', f=frame_size)
                sim = quant @ global_latent.transpose(-1, -2)

                r = int(quant.shape[1] * 0.8)
                max_value, max_idx = sim.max(dim=-1) #  b (f h w)
                max_idx_sorted = max_value.argsort(dim=-1, descending=True)[..., None] # b (f h w) 1
                quant_replaced_idx = max_idx_sorted[..., :r, :]  # Merged Tokens
                dict_idx = torch.gather(max_idx[..., None], dim=-2, index=quant_replaced_idx)

                # 假设global_latent的形状为 [b, n, c]，其中n是特征数量，c是每个特征的维度
                # dict_idx的形状为 [b, r, 1]，其中r是您选择的特征数量

                # 使用torch.gather从global_latent中按照dict_idx提取特征
                selected_features = torch.gather(global_latent, 1, dict_idx.expand(-1, -1, global_latent.size(-1)))

                # quant_replaced_idx的形状应该调整为 [b, r, 1]，与selected_features匹配
                # 确保quant的第二个维度是展开的形状 (b, f*h*w, c)，以便于索引操作

                # 将提取的特征放置到quant中相应的位置
                quant = quant.view(-1, quant.shape[-1])  # 展开quant
                quant_replaced_idx = quant_replaced_idx.view(-1, 1)  # 展开quant_replaced_idx为适当的形状进行索引
                quant.scatter_(0, quant_replaced_idx, selected_features.view(-1, selected_features.shape[-1]))

                # 重新调整quant的形状回原始形状
                quant = quant.view(-1, frame_size*height*width, quant.shape[-1])  # b, f*h*w, c
                global_latent = quant
                quant = rearrange(quant, 'b (f h w) c -> b f c h w', f=frame_size, h=height, w=width)


                latents.append(quant)

        video = []
        for latent in latents:
            batch_size, frame_size = latent.shape[0], latent.shape[1]
            image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
            latent = rearrange(latent, 'b f c h w -> (b f) c h w')
            video_clip = self.generator(latent, image_only_indicator=image_only_indicator)
            video_clip = rearrange(video_clip, '(b f) c h w -> b f c h w', f=frame_size)
            video.append(video_clip)
        video = torch.cat(video, dim=1)
        return video

@ARCH_REGISTRY.register()
class TemporalVQVAEV2(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None, freeze_backbone=True, training_module_names=[], use_grad_checkpoint=True, motion_bank_size=4096):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = VideoGeneratorV2(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'], strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'],strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')
            
        # freeze backbone
        if freeze_backbone:
            for name, p in self.named_parameters():
                p.requires_grad_(False)
                for each in training_module_names:
                    if each in name:
                        p.requires_grad_(True)


    def forward(self, x):
        batch_size, frame_size, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = self.generator(quant.requires_grad_(True), image_only_indicator=image_only_indicator)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        return x, codebook_loss, quant_stats
    
@ARCH_REGISTRY.register()
class TemporalVQVAEV2Motion(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None, freeze_backbone=True, training_module_names=[], use_grad_checkpoint=True, motion_bank_size=4096):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
            if motion_bank_size is not None:
                self.motion_quantize = VectorQuantizer(motion_bank_size, 16, self.beta)
            else:
                self.motion_quantize = VectorQuantizer(self.codebook_size*16, 16, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = VideoGeneratorV2(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'], strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'],strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')
            
        # freeze backbone
        if freeze_backbone:
            for name, p in self.named_parameters():
                p.requires_grad_(False)
                for each in training_module_names:
                    if each in name:
                        p.requires_grad_(True)

    def get_quant_motion(self, x):
        assert x.dim() == 5, 'shape must be (b f c h w)'
        quant_motion_mean = torch.mean(x, dim=2, keepdim=True)
        quant_motion_var = torch.var(x, dim=2, keepdim=True)
        quant_motion = torch.cat([quant_motion_mean, quant_motion_var], dim=2)
        quant_motion = rearrange(quant_motion, 'b f c h w -> b (f c) h w')
        quant_motion, quant_loss, _ = self.motion_quantize(quant_motion)
        return quant_motion, quant_loss

    def modulate_motion(self, x, motion_mean_var, eps=1e-6):
        assert x.dim() == 5, 'shape must be (b f c h w)'
        assert motion_mean_var.dim() == 4, 'shape must be (b (2f) h w)'
        motion_mean = torch.mean(x, dim=2, keepdim=True)
        motion_var = torch.var(x, dim=2, keepdim=True)

        motion_mean_var = rearrange(motion_mean_var, 'b (f c) h w -> b f c h w', c=2)
        new_mean, new_var = motion_mean_var.chunk(2, dim=2)
        x = (x - motion_mean) / torch.sqrt(motion_var + eps)
        x = x * torch.sqrt(new_var) + new_mean

        return x


    def forward(self, x):
        batch_size, frame_size, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        quant_motion = rearrange(quant, '(b f) c h w -> b f c h w', f=frame_size)
        quant_motion, quant_motion_loss = self.get_quant_motion(quant_motion)
        quant = rearrange(quant, '(b f) c h w -> b f c h w', f=frame_size)
        quant = self.modulate_motion(quant, quant_motion)
        quant = rearrange(quant, 'b f c h w -> (b f) c h w', f=frame_size)

        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = self.generator(quant, image_only_indicator=image_only_indicator)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        return x, (codebook_loss, quant_motion_loss), quant_stats

@ARCH_REGISTRY.register()
class TemporalVQVAEv0(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=[16], codebook_size=1024, emb_dim=256,
                beta=0.25, gumbel_straight_through=False, gumbel_kl_weight=1e-8, model_path=None, freeze_backbone=True, training_module_names=[], use_grad_checkpoint=True):
        super().__init__()
        logger = get_root_logger()
        self.in_channels = 3 
        self.nf = nf 
        self.n_blocks = res_blocks 
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels,
            self.nf,
            self.embed_dim,
            self.ch_mult,
            self.n_blocks,
            self.resolution,
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint
        )
        if self.quantizer_type == "nearest":
            self.beta = beta #0.25
            self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        elif self.quantizer_type == "gumbel":
            self.gumbel_num_hiddens = emb_dim
            self.straight_through = gumbel_straight_through
            self.kl_weight = gumbel_kl_weight
            self.quantize = GumbelQuantizer(
                self.codebook_size,
                self.embed_dim,
                self.gumbel_num_hiddens,
                self.straight_through,
                self.kl_weight
            )
        self.generator = VideoGenerator(
            self.nf, 
            self.embed_dim,
            self.ch_mult, 
            self.n_blocks, 
            self.resolution, 
            self.attn_resolutions,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_ema' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_ema'], strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params_ema]')
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'],strict=False)
                logger.info(f'vqgan is loaded from: {model_path} [params]')
            else:
                raise ValueError(f'Wrong params!')
            
        # freeze backbone
        if freeze_backbone:
            for name, p in self.named_parameters():
                p.requires_grad_(False)
                for each in training_module_names:
                    if each in name:
                        p.requires_grad_(True)


    def forward(self, x):
        batch_size, frame_size, _, _, _ = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.encoder(x)
        quant, codebook_loss, quant_stats = self.quantize(x)
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = self.generator(quant, image_only_indicator=image_only_indicator)
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        return x, codebook_loss, quant_stats



# patch based discriminator
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.ModuleList(layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        for block in self.main:
            x = block(x)
        return x
    

@ARCH_REGISTRY.register()
class VQGAN3DDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        return self.main(x)