import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

@torch.no_grad()
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSAVisualLayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # tgt (fhw)bc

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, atten = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        torch.save(atten, 'atten_map.pt')
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # tgt (fhw)bc

        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class TransformerCALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, q, k, v,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                keyvalue_pos: Optional[Tensor] = None):
        # tgt (fhw)bc
        tgt = q
        # self attention
        q = self.norm1(q)
        k = self.norm1(k)
        v = self.norm1(v)
        q = self.with_pos_embed(q, query_pos)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class TransformerSpatialTemporalLayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = TransformerSpatioTemporalModelSon(num_attention_heads=nhead,
                                                           attention_head_dim=embed_dim // nhead, in_channels=embed_dim)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # tgt (fhw)bc
        batch_size, frame_size, _, height, width = tgt.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(tgt.device)

        tgt = rearrange(tgt, 'b f c h w -> (h w) (b f) c')

        # self attention
        # tgt (fhw)bc
        tgt2 = self.norm1(tgt)
        qkv = self.with_pos_embed(tgt2, query_pos)
        qkv = rearrange(qkv, '(h w) (b f) c -> (b f) c h w', f=frame_size, h=height, w=width, b=batch_size)
        tgt2 = self.self_attn(qkv, image_only_indicator=image_only_indicator)
        tgt2 = rearrange(tgt2, '(b f) c h w -> (h w) (b f) c', b=batch_size, f=frame_size)
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = rearrange(tgt, "(h w) (b f) c -> b f c h w", b=batch_size, f=frame_size, h=height, w=width)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)

        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        
        self.zero_conv = zero_module(nn.Conv2d(out_ch,out_ch,1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        # scale = self.scale(enc_feat)
        # shift = self.shift(enc_feat)
        # enc_feat = checkpoint(self.encode_enc, torch.cat([enc_feat, dec_feat], dim=1))
        scale = checkpoint(self.scale, enc_feat)
        shift = checkpoint(self.shift, enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + self.zero_conv(residual)
        return out


@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None):
        super(CodeFormer, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (hw)bn
        logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:  # fuse after i-th block
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat


@ARCH_REGISTRY.register()
class CodeFormerV2(VQAutoEncoder):
    """
    Replace self-attention with cross-attention
    Q from degraded feature
    K from dictionary elements
    V from dictionary elements & add one-hot position embedding
    """

    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None):
        super(CodeFormerV2, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'])

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.dict_to_emb = nn.Linear(latent_size, self.dim_embd)
        self.onehot_emb = nn.Linear(codebook_size, self.dim_embd)
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerCALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb
        # get dictionary feature
        dict_emb = self.dict_to_emb(self.quantize.embedding.weight)
        dict_emb = dict_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        # one-hot position embedding
        onehot_emb = self.onehot_emb(torch.eye(self.quantize.codebook_size).to(x.device))
        onehot_emb = onehot_emb.unsqueeze(1).repeat(1, x.shape[0], 1)

        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(q=query_emb, k=dict_emb, v=onehot_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (hw)bn
        logits = logits.permute(1, 0, 2)  # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]

        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:  # fuse after i-th block
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat


@ARCH_REGISTRY.register()
class VideoCodeFormer(TemporalVQVAE):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256, st_latent_size=None,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None):
        super(VideoCodeFormer, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size,
                                              freeze_backbone=False)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'], strict=False)

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))  # (f h w) c
        self.st_position_emb = nn.Parameter(
            torch.zeros(st_latent_size, self.dim_embd)) if st_latent_size else None  # (f h w) c
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        # for f_size in self.connect_list:
        #     in_ch = self.channels[f_size]
        #     self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                module_ = getattr(self, module, None)
                if module is None:
                    print(f"No module name {module_}")
                    continue
                module = module_
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_spatialtemporal_position_weight(self, temporal_size=1):
        if self.st_position_emb is not None:
            self.st_position_emb.data.copy_(self.position_emb.data.repeat(temporal_size, 1))
            
    def debug_mean_var(self, x, name):
        # mean = torch.mean(x,dim=2) # b f h w
        # var = torch.var(x,dim=2) # b f h w
        
        # # m_mean = torch.mean(mean, dim=1)
        # m_var = torch.var(mean, dim=1).sqrt()
        torch.save(x, f"{name}_feat.pt")
        # torch.save(m_var, f"{name}_var.pt")
        # print(name)
        # var = torch.var(x, dim=1).mean(dim=1).mean().item() # b c h w
        # with open(f"log_{name}",'a') as f:
        #     f.write(f"{var}\n")

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # self.debug_mean_var(lq_feat, 'HQ')
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            
        quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # self.debug_mean_var(quant_feat, 'HQ_bank1')
        quant_motion, _ = self.get_quant_motion(quant_feat)

        quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # self.debug_mean_var(quant_feat_modulated, 'bank2')
        quant_feat = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            x = checkpoint(block, x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    
@ARCH_REGISTRY.register()
class VideoCodeFormerInpainting(TemporalVQVAEv0):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256, st_latent_size=None,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None):
        super(VideoCodeFormerInpainting, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size,
                                              freeze_backbone=False)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'], strict=False)

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))  # (f h w) c
        self.st_position_emb = nn.Parameter(
            torch.zeros(st_latent_size, self.dim_embd)) if st_latent_size else None  # (f h w) c
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                module_ = getattr(self, module, None)
                if module is None:
                    print(f"No module name {module_}")
                    continue
                module = module_
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_spatialtemporal_position_weight(self, temporal_size=1):
        if self.st_position_emb is not None:
            self.st_position_emb.data.copy_(self.position_emb.data.repeat(temporal_size, 1))
            

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # self.debug_mean_var(lq_feat, 'HQ')
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(frame_size, batch_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # # self.debug_mean_var(quant_feat, 'HQ_bank1')
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # # self.debug_mean_var(quant_feat_modulated, 'bank2')
        # quant_feat = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            f_size = str(x.shape[-1])
            
            if int(f_size)<=1:
                x = checkpoint(block, x)
                if isinstance(block, ResBlock):
                    block_ = temporal_blocks_iter.pop()
                    x = checkpoint(block_, x, None, image_only_indicator)
                elif isinstance(block, AttnBlock):
                    block_ = temporal_attens_iter.pop()
                    x = checkpoint(block_, x, None, image_only_indicator)
            else:
                x = block(x)
                if isinstance(block, ResBlock):
                    block_ = temporal_blocks_iter.pop()
                    x = block_(x, None, image_only_indicator)
                elif isinstance(block, AttnBlock):
                    block_ = temporal_attens_iter.pop()
                    x = block_(x, None, image_only_indicator)
                    
            if i in fuse_list:  # fuse after i-th block
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat

@ARCH_REGISTRY.register()
class VideoCodeFormerStage2p5(VideoCodeFormer):
    def __init__(self,
                n_head: int = 8,
                *args, **kwargs):
        super().__init__(
            n_head=n_head,
            *args, **kwargs)

        self.bank_crossAtten = nn.Sequential(
            *[TransformerCALayer(embed_dim=self.embed_dim, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers//2)])


    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        quant_motion, _ = self.get_quant_motion(quant_feat)

        quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        quant_feat_modulated = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
            quant_feat_modulated = quant_feat_modulated.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        quant_feat_modulated = rearrange(quant_feat_modulated, '(b f) c h w -> (f h w) b c', f=frame_size, b=batch_size)
        quant_feat = rearrange(quant_feat, 'b f c h w -> (f h w) b c')

        for layer in self.bank_crossAtten:
            quant_feat = layer(q=quant_feat, k=quant_feat_modulated, v=quant_feat_modulated)
        quant_feat = rearrange(quant_feat, '(f h w) b c -> (b f) c h w', f=frame_size, h=16, w=16)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            if block != self.generator.blocks[-1]:
                x = checkpoint(block, x)
            else:
                x = block(x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    
@ARCH_REGISTRY.register()
class VideoCodeFormerStage3(VideoCodeFormerStage2p5):
    def __init__(self, fix_modules=None, *args, **kwargs):
        super().__init__(fix_modules=None, *args, **kwargs)
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)
            

        if fix_modules is not None:
            for module in fix_modules:
                module = getattr(self, module)
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False
    
    def debug_mean_var(self, x, name):
        # mean = torch.mean(x,dim=2) # b f h w
        # var = torch.var(x,dim=2) # b f h w
        
        # m_mean = torch.mean(mean, dim=1)
        # m_var = torch.var(mean, dim=1).sqrt()
        # torch.save(x, f"{name}_feat.pt")
        # torch.save(m_var, f"{name}_var.pt")
        x = x.detach()
        dis = 0
        for i in range(1,x.shape[1]):
            dis += (x[:,i] - x[:,i-1]).square().sqrt().mean()
        dis = dis / 8
        # print(dis)
        with open(f"delta_{name}", 'a') as f:
            f.write(str(dis.item())+'\n')
        # torch.save(x, f"{name}_feat.pt")
        # torch.save(dis, f"{name}_delta.pt")
        
    
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        print("LQ:")
        self.debug_mean_var(lq_feat,"HQ")
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        print("bank1:")
        self.debug_mean_var(quant_feat, "HQbank1")
        quant_motion, _ = self.get_quant_motion(quant_feat)

        quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # print("bank2:")
        # self.debug_mean_var(quant_feat_modulated,"bank2")
        quant_feat_modulated = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
            quant_feat_modulated = quant_feat_modulated.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        quant_feat_modulated = rearrange(quant_feat_modulated, '(b f) c h w -> (f h w) b c', f=frame_size, b=batch_size)
        quant_feat = rearrange(quant_feat, 'b f c h w -> (f h w) b c')

        for layer in self.bank_crossAtten:
            quant_feat = checkpoint(layer, quant_feat, quant_feat_modulated, quant_feat_modulated)
            # quant_feat = layer(q=quant_feat, k=quant_feat_modulated, v=quant_feat_modulated)
        quant_feat = rearrange(quant_feat, '(f h w) b c -> (b f) c h w', f=frame_size, h=16, w=16)
        # print("CA:")
        # self.debug_mean_var(rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size),"CA")

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            # if block not in self.generator.blocks[-3:]:
            #     x = checkpoint(block, x)
            # else:
            #     x = block(x)
            x = block(x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                # x = checkpoint(block_, x, None, image_only_indicator)
                x = block_(x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
                
            if i in fuse_list: # fuse after i-th block
                f_size = str(x.shape[-1])
                if w >= 0:
                    block_ = self.fuse_convs_dict[f_size]
                    x = block_(enc_feat_dict[f_size].detach(), x, w)
                    # if f_size == self.connect_list[-1]:
                    #     x = block_(enc_feat_dict[f_size].detach(), x, w)
                    # else:
                    #     x = checkpoint(block_, enc_feat_dict[f_size].detach(), x, w)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    
@ARCH_REGISTRY.register()
class VideoCodeFormerSAware(VideoCodeFormer):
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # self.debug_mean_var(lq_feat, 'HQ')
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (h w) (b f) c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # hw(bf)n
        logits = rearrange(logits, 'a b c -> b a c')  # (hw)(bf)n -> (bf)(hw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # # self.debug_mean_var(quant_feat, 'HQ_bank1')
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # # self.debug_mean_var(quant_feat_modulated, 'bank2')
        # quant_feat = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            x = checkpoint(block, x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    


@ARCH_REGISTRY.register()
class VideoCodeFormerV2(TemporalVQVAEV2):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256, st_latent_size=None,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None):
        super(VideoCodeFormerV2, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size,
                                              freeze_backbone=False)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'], strict=False)

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))  # (f h w) c
        self.st_position_emb = nn.Parameter(
            torch.zeros(st_latent_size, self.dim_embd)) if st_latent_size else None  # (f h w) c
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        # for f_size in self.connect_list:
        #     in_ch = self.channels[f_size]
        #     self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                module_ = getattr(self, module, None)
                if module is None:
                    print(f"No module name {module_}")
                    continue
                module = module_
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_spatialtemporal_position_weight(self, temporal_size=1):
        if self.st_position_emb is not None:
            self.st_position_emb.data.copy_(self.position_emb.data.repeat(temporal_size, 1))
            
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # self.debug_mean_var(lq_feat, 'HQ')
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            x = checkpoint(block, x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    
@ARCH_REGISTRY.register()
class VideoCodeFormerV2Motion(TemporalVQVAEV2Motion):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9,
                 codebook_size=1024, latent_size=256, st_latent_size=None,
                 connect_list=['32', '64', '128', '256'],
                 fix_modules=['quantize', 'generator'], vqgan_path=None,motion_bank_size=4096):
        super(VideoCodeFormerV2Motion, self).__init__(512, 64, [1, 2, 2, 4, 4, 8], 'nearest', 2, [16], codebook_size,
                                              freeze_backbone=False, motion_bank_size=motion_bank_size)

        if vqgan_path is not None:
            self.load_state_dict(
                torch.load(vqgan_path, map_location='cpu')['params_ema'], strict=False)

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))  # (f h w) c
        self.st_position_emb = nn.Parameter(
            torch.zeros(st_latent_size, self.dim_embd)) if st_latent_size else None  # (f h w) c
        self.feat_emb = nn.Linear(256, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 256,
            '64': 256,
            '128': 128,
            '256': 128,
            '512': 64,
        }

        # after second residual block for > 16, before attn layer for ==16
        self.fuse_encoder_block = {'512': 2, '256': 5, '128': 8, '64': 11, '32': 14, '16': 18}
        # after first residual block for > 16, before attn layer for ==16
        self.fuse_generator_block = {'16': 6, '32': 9, '64': 12, '128': 15, '256': 18, '512': 21}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        # for f_size in self.connect_list:
        #     in_ch = self.channels[f_size]
        #     self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        if fix_modules is not None:
            for module in fix_modules:
                module_ = getattr(self, module, None)
                if module is None:
                    print(f"No module name {module_}")
                    continue
                module = module_
                if isinstance(module, nn.Parameter):
                    module.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_spatialtemporal_position_weight(self, temporal_size=1):
        if self.st_position_emb is not None:
            self.st_position_emb.data.copy_(self.position_emb.data.repeat(temporal_size, 1))
            
    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # self.debug_mean_var(lq_feat, 'HQ')
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
            
        # quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        # quant_motion, _ = self.get_quant_motion(quant_feat)

        # quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        # quant_feat = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            x = checkpoint(block, x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat
    
@ARCH_REGISTRY.register()
class VideoCodeFormerStage2p5V2(VideoCodeFormerV2Motion):
    def __init__(self,
                n_head: int = 8,
                *args, **kwargs):
        super().__init__(
            n_head=n_head,
            *args, **kwargs)

        self.bank_crossAtten = nn.Sequential(
            *[TransformerCALayer(embed_dim=self.embed_dim, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0)
              for _ in range(self.n_layers//2)])


    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False, consistency_loss=False):
        # ################### Encoder #####################
        batch_size, frame_size, _, _, _ = x.shape
        image_only_indicator = torch.zeros((batch_size, frame_size)).to(x.device)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        x = rearrange(x, '(b f) c h w -> b f c h w', f=frame_size)
        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        if self.st_position_emb is None:
            pos_emb = self.position_emb.unsqueeze(1).repeat(1, batch_size * frame_size, 1)
        else:
            pos_emb = self.st_position_emb.unsqueeze(1).repeat(1, batch_size, 1)
        # BFCHW -> (FHW)BC
        lq_feat_flatten = rearrange(lq_feat, 'b f c h w -> (f h w) b c')
        feat_emb = self.feat_emb(lq_feat_flatten)
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        consist_loss = None

        # output logits
        logits = self.idx_pred_layer(query_emb)  # (fhw)bn
        logits = rearrange(logits, 'a b c -> b a c')  # (fhw)bn -> b(fhw)n
        # logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only:  # for training stage II
            # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat, consist_loss

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx,
                                                     shape=[int(batch_size * frame_size), 16, 16, 256])  # (bf) c h w
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()
        quant_feat = rearrange(quant_feat, '(b f) c h w -> b f c h w', f=frame_size)
        quant_motion, _ = self.get_quant_motion(quant_feat)

        quant_feat_modulated = self.modulate_motion(quant_feat, quant_motion)
        quant_feat_modulated = rearrange(quant_feat_modulated, 'b f c h w -> (b f) c h w', f=frame_size)

        if detach_16:
            quant_feat = quant_feat.detach()  # for training stage III
            quant_feat_modulated = quant_feat_modulated.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        quant_feat_modulated = rearrange(quant_feat_modulated, '(b f) c h w -> (f h w) b c', f=frame_size, b=batch_size)
        quant_feat = rearrange(quant_feat, 'b f c h w -> (f h w) b c')

        for layer in self.bank_crossAtten:
            quant_feat = layer(q=quant_feat, k=quant_feat_modulated, v=quant_feat_modulated)
        quant_feat = rearrange(quant_feat, '(f h w) b c -> (b f) c h w', f=frame_size, h=16, w=16)

        # ################## Generator ####################
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        temporal_blocks_iter = list(self.generator.temporal_blocks)[::-1]
        temporal_attens_iter = list(self.generator.temporal_attens)[::-1]
        for i, block in enumerate(self.generator.blocks):
            if block != self.generator.blocks[-1]:
                x = checkpoint(block, x)
            else:
                x = block(x)
            if isinstance(block, ResBlock):
                block_ = temporal_blocks_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
            elif isinstance(block, AttnBlock):
                block_ = temporal_attens_iter.pop()
                x = checkpoint(block_, x, None, image_only_indicator)
        out = x
        # logits doesn't need softmax before cross_entropy loss
        out = rearrange(out, '(b f) c h w -> b f c h w', f=frame_size)
        # logits: b(fhw)n
        # lq_feat: bfchw

        return out, logits, lq_feat