import math
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import os

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

#### CONSTANTS
DEVICE = xm.xla_device()

#### TESTS
def test_res():
    x = torch.randn((1, 128, 32, 32))
    res = ResnetBlock(128, 256, temb=False)
    y = res(x)
    assert y.shape == (1, 256, 32, 32)

def test_sin():
    sin = SinusoidalEmbedding(50, 256)
    assert sin[0].shape == (256,)

def test_res_temb():
    temb = SinusoidalEmbedding(50, 256)
    x = gpu(torch.randn((1, 128, 32, 32)))
    res = gpu(ResnetBlock(128, 256, emb_channels=256))
    y = res(x, temb(0))
    assert y.shape == (1, 256, 32, 32)

def test_attn():
    x = torch.randn((1, 128, 32, 32))
    attn = AttnBlock(128)
    y = attn(x)
    assert y.shape == (1, 128, 32, 32)

def test_unet():
    x = gpu(torch.randn((1, 3, 32, 32)))
    u = gpu(UNet(128))
    t = gpu(torch.tensor(2))
    y = u(x, t)
    #print(y.shape)

def test_time_emb():
    x = gpu(torch.tensor(5))
    t = SinusoidalEmbedding(50, 128)

def gpu(x):
    return x.to(DEVICE)

def test_data_loader():

    BATCH_SIZE = 32

    ds = torchvision.datasets.CIFAR10(download=True, root=".")
    diffusion_ds = remove_labels(ds)
    diffusion_ds = totensor_ds(diffusion_ds)
    dl = BatchedDataLoader(diffusion_ds, BATCH_SIZE)
    for i in range(len(dl)):
        dl[i]
    dl[0]
    dl[-1]

def test_diffusion():
    UNET_DIM = 128
    BATCH_SIZE = 128

    ds = torchvision.datasets.CIFAR10(download=True, root=".")
    diffusion_ds = remove_labels(ds)
    diffusion_ds = totensor_ds(diffusion_ds)
    dl = BatchedDataLoader(diffusion_ds, batch_size=BATCH_SIZE)
    eps_model = UNet(UNET_DIM)
    eps_model = gpu(eps_model)
    diffusion = DiffusionModel(eps_model)
    diffusion = gpu(diffusion)
    diffusion._sample()
    diffusion._train(dl, epochs=1, train_steps_per_epoch=-1, batch_size = BATCH_SIZE)

def Normalize(in_channels, num_groups=2):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size, num_hidden_units):
        super(SinusoidalEmbedding, self).__init__()
        PE = torch.zeros((size, num_hidden_units))
        for i in range(size):
            for j in range(num_hidden_units):
                if i % 2 == 0:
                    PE[i, j] = math.sin(i/ 10000 ** (2*j/num_hidden_units))
                else:
                    PE[i, j] = math.cos(i/10000**(2*j/num_hidden_units))
        self.PE = PE
        self.PE = gpu(self.PE)
    def __getitem__(self, i):
        return self.PE[i]
    def forward(self, i):
        return self.PE[i]

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        h = x
        x = F.relu(self.conv2(x))
        x = x + h

class ResnetBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        dropout=0.1,
        emb_channels=512,
        temb=True,
        tembs=None
        ):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = False

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(out_channels)

        if temb:
            self.temb = True
            self.linear = torch.nn.Linear(emb_channels, out_channels)
        else:
            self.temb = False

        self.emb_proj = torch.nn.Linear(emb_channels, out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.resize_shortcut = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)

        self.c = None

    def forward(self, x, emb=None):
        batch_size = x.shape[0]
        x_prev = x
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)

        if self.temb:
            emb = self.linear(emb)
            x += emb.reshape(batch_size, -1, 1, 1)

        x = self.norm2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x_prev = self.conv_shortcut(x_prev)
            else:
                x_prev = self.resize_shortcut(x_prev)

        return x + x_prev


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.c = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape

        if self.c == None:
            self.c = torch.tensor(c, device=DEVICE)

        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)    # b,hw,c
        k = k.reshape(b,c,h*w)  # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * torch.pow(self.c, 0.5) # using torch.sqrt breaks tpu
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
##

import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

##
class bUNet(nn.Module):
    def __init__(
        self,
        num_channels,
        channel_mult = (1, 2, 4, 8),
        ch=4,
        out_ch=3,
        num_res_blocks=1
        ):
        super(UNet, self).__init__()

        # constants
        self.num_timesteps = 50
        self.num_channels = num_channels
        self.temb_ch = self.num_channels *4
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_mult)

        # fancy constants
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        channel_scale = [self.num_channels * i for i in channel_mult]

        self.up_scales = list(zip(
            map(lambda x: x*2, channel_scale[::-1]), channel_scale[::-1]))
        self.down_scales = list(zip(
            [channel_scale[0], *channel_scale][:-1], channel_scale
        ))

        self.in_conv = nn.Conv2d(3, self.num_channels, kernel_size=7, padding="same")

        self.time_mlp = torch.nn.Sequential(
            SinusoidalEmbedding(self.num_timesteps, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch),
            nn.SiLU()
        )

        # down
        for i, dims in enumerate(self.down_scales): #range(len(channel_mult)):
            in_dim, out_dim = dims

            for block in range(num_res_blocks):
                res1 = ResnetBlock(in_dim, out_dim, emb_channels=self.temb_ch)
                res2 = ResnetBlock(out_dim, out_dim, emb_channels=self.temb_ch)
                attn = AttnBlock(out_dim)

                down_block = torch.nn.Sequential(
                    res1,
                    res2,
                    attn
                )

                self.down.append(down_block)

            if i != len(channel_mult) - 1:
                self.down.append(torch.nn.MaxPool2d(2, 2))

        prev_dim = self.down_scales[-1][-1]

        self.mid = torch.nn.Sequential(
            ResnetBlock(prev_dim, prev_dim * 4, temb=False),
            AttnBlock(prev_dim * 4),
            ResnetBlock(prev_dim * 4, prev_dim, temb=False),
        )

        # up part of unet
        prev_dim=prev_dim + prev_dim

        for i, dims in enumerate(self.up_scales): #range(len(channel_mult)):
            in_dim, out_dim = dims

            for block in range(num_res_blocks):
                res1 = ResnetBlock(in_dim, out_dim, emb_channels=self.temb_ch)
                res2 = ResnetBlock(out_dim, out_dim, emb_channels=self.temb_ch)
                attn = AttnBlock(out_dim)

                up_block = torch.nn.Sequential(
                    res1,
                    res2,
                    attn
                )

                self.up.append(up_block)

            if i != self.num_resolutions - 1:
                self.up.append(torch.nn.ConvTranspose2d(out_dim, out_dim // 2, 3, stride=2, padding=1, output_padding=1))

        last_dim = self.up_scales[-1][-1]
        self.out_conv = nn.Conv2d(last_dim, 3, stride=1, kernel_size=3, padding=1)

    def forward(self, x, t=None):

        if t is not None:
            t_emb = self.time_mlp(t)

        x = self.in_conv(x)

        state = []
        for i, module in enumerate(self.down):
            if type(module) != nn.MaxPool2d:
                res1, res2, attn = module
                x = res1(x, t_emb)
                x = res2(x, t_emb)
                x = attn(x)
            else:
                # if this layer downsamples, we will add x to state
                state.append(x)
                x = module(x)

        state.append(x)

        x = self.mid(x)
        add_l = False

        l = state.pop()
        x = torch.cat((x, l), 1)

        for module in self.up:
            if type(module) != nn.ConvTranspose2d:
                res1, res2, attn = module
                x = res1(x, t_emb)
                x = res2(x, t_emb)
                x = attn(x)
            else:
                # if this layer upsamples, we will pop off of state
                x = module(x)
                l = state.pop()
                x = torch.cat((x, l), 1)

        x = self.out_conv(x)

        return x

    def count_parameters(self):
        tot=0
        for param in self.parameters():
            tot += param.numel()
        print(tot)

    def dump_state_dict(self, filename='state_dict.txt'):
        for k, v in self.state_dict().items():
            print(f'{k} {v.shape}')

# HELPER FUNCTIONS
def remove_labels(ds):
    return list(map(lambda x: x[0], ds))

def totensor_ds(ds):
    t = transforms.ToTensor()
    return list(map(lambda x: t(x), ds))

def make_diffusion_dataset(ds):
    im = []
    t = torchvision.transforms.ToTensor()
    for img, label in ds:
        im.append(t(img))
    im = im[:10]
    for i, img in enumerate(im):
        noise_list = noise(img)
        im[i] = [img]
        #[im[i].append(a) for a in noise_list]
    return im

# perform a sequence of noising steps on an image. Then, generalize it to n noising steps (Apply combined product)
def noise(img):
    beta = 0.4
    sqrt_beta = math.sqrt(1 - beta)
    img = img * torch.normal(img * sqrt_beta, beta)
    return img

# noise an image for n steps
def noise_n_steps(img, betas, n):
    img = torch.clone(img)

    for b, i in zip(betas, range(n)):
        beta = b
        sqrt_beta = (1 - beta) ** 0.5
        img *= torch.normal(img * sqrt_beta, beta) #torch.ones(img.shape) * sqrt_beta, beta)
    return img

# noise an image for 1 step, given beta
def noise_1_step(img, beta):
    sqrt_beta = math.sqrt(1 - beta)
    img *= torch.normal(torch.ones(img.shape) * sqrt_beta, beta)

# closed form sampling for the noise using alpha param
def noise_nth_step(img, betas, t):
    alphas_cumprod = 1
    for i in range(t):
        alpha = 1 - betas[i]
        alphas_cumprod *= alpha
    return torch.normal(img * math.sqrt(alphas_cumprod), 1 - alphas_cumprod)

def noise_alpha(img, betas, t):
    alpha = torch.prod(1 - betas[:t])
    return torch.normal(img * (alpha ** 0.5), (1 - alpha))

# single loss update using KL divergence
# KL is the distance between probability of distributions p and q
def loss_update(q_t0, p_xt, p_01):
    return

# calculate alpha at a certain timestep t
def calculate_alpha(betas, t):
    tensor = torch.zeros_like(t)
    return tensor
    for i in range(tensor.shape[0]):
        tensor[i] = torch.prod(1 - betas[:t[i]])

    return tensor

def linear_beta_schedule(num_timesteps):
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)

def fake_normalize(t: torch.Tensor):
    return (t - torch.mean(t)) / torch.var(t)

class BatchedDataLoader():
    def __init__(self, ds, batch_size=32):
        self.len = len(ds)
        self.ds = torch.stack(ds)
        self.batch_size =batch_size
    def set_bsize(self, new_batch_size):
        self.batch_size =new_batch_size
    def __len__(self):
        return self.len
        #return len(self.ds) // self.batch_size
    def __getitem__(self, i):
        i *= self.batch_size
        return self.ds[i:i+self.batch_size]

class DiffusionModel(nn.Module):
    def __init__(self, eps_model, epochs=10, train_steps_per_epoch=-1, beta_schedule=None):
        super(DiffusionModel, self).__init__()
        self.epochs = epochs
        self.train_steps = train_steps_per_epoch # -1 just means don't interrupt
        self.cum_loss = []
        self.eps_model = eps_model

        if beta_schedule is not None:
            self.beta_schedule = beta_schedule
        else:
            self.beta_schedule = linear_beta_schedule

    def log(self, n_iter, n_epoch, loss):
        self.cum_loss.append((n_iter, n_epoch, float(loss)))
        print(f'LOSS LOG: {loss}')
        pass

    def log_print(self, msg):
        print(f'PRINT LOG: {msg}')

    # TRAINING LOOP
    def _train(self, dataset, epochs=10, train_steps_per_epoch=-1, n_noise_steps=50, batch_size=32):
        # debug
        #torch.autograd.set_detect_anomaly(True)
        self.log_print("beginning training")
        betas = linear_beta_schedule(n_noise_steps)
        optim = torch.optim.AdamW(self.eps_model.parameters(), lr=2e-5) # we leave default params
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            for batch, img in enumerate(dataset):
                if train_steps_per_epoch != -1:
                    if batch == train_steps_per_epoch:
                        print('we return here')
                        return
                img = dataset[8]
                img = gpu(img)
                print('translating')

                t = torch.randint(0, n_noise_steps, (batch_size,))
                t = gpu(t)
                print('gpud')

                x_0 = img

                # batched_calculate_alpha
                alpha = gpu(calculate_alpha(betas, t))
                print('calculate alpha')
                alpha = alpha.reshape(-1, 1, 1, 1)

                sqrt_alpha = torch.pow(1 - alpha, 0.5) #using torch.sqrt breaks tpu for some reason

                eps = torch.randn_like(x_0, requires_grad=True)
                eps = gpu(eps)

                eps_out = self.eps_model((alpha) * x_0 + (sqrt_alpha) * eps, t)
                print('model out')

                loss = loss_fn(eps, eps_out)
                print('loss')
                loss.backward()
                print('backward')

                optim.step()
                xm.mark_step()
                print('optim step')

                #print(loss)

                report=met.metrics_report().split('\n')
                print(torch_xla._XLAC._xla_metrics_report())

                #self.log(batch, epoch, loss)

    def _sample(self, n_noise_steps=50):
        img_shape = (1, 3, 32, 32)
        betas = linear_beta_schedule(n_noise_steps)
        x = torch.normal(torch.zeros(img_shape), 1)
        for t in range(0, n_noise_steps, -1):
            sig = alpha ** 2
            x = (1 / alpha) * (x - ((1 - alpha) / ((1 - alpha) ** 2)) * self.eps_model(x, t))

        return x

if __name__ == "__main__":

    #test_res()
    #test_res_temb()
    #test_sin()
    #test_attn()
    #test_unet()
    #test_time_emb()
    os.environ['XLA_IR_DEBUG'] = '1'
    test_diffusion()
    #test_data_loader()
    #print('device loaded successfully')
    #print(DEVICE)