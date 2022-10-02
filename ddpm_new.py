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
    x = torch.randn((1, 128, 32, 32))
    res = ResnetBlock(128, 256)
    y = res(x, temb[0])
    assert y.shape == (1, 256, 32, 32)

def test_attn():
    x = torch.randn((1, 128, 32, 32))
    attn = AttnBlock(128)
    y = attn(x)
    assert y.shape == (1, 128, 32, 32)

def test_unet():
    x = torch.randn((1, 3, 32, 32))
    u = UNet(64, 64)
    y = u(x)

def Normalize(in_channels, num_groups=2):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

"""
PE(pos,2i) = sin(pos/10000**(2*i/hidden_units))
PE(pos,2i+1) = cos(pos/10000**(2*i/hidden_units))
todo make not shitty
"""

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
    def __getitem__(self, i):
        return self.PE[i]

"""
class RandomFourierFeatures(nn.Module):
    def __init__(self, size):
        super(TimestepEmbedding, self).__init__()
        beta = 
        alpha = 0
        self.temb_block = []
        for i in temb_block:
            pass
    def forward(self, i):
        return self.temb_block[i]
"""    

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

        temb = False

        # can just do linear for timestep embedding
        if temb:
            self.temb = True
            self.linear = torch.nn.Linear(out_channels, out_channels)
        else:
            self.temb = False

        self.emb_proj = torch.nn.Linear(emb_channels, out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.resize_shortcut = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x, emb=None):

        x_prev = x
        #x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)

        if self.temb:
            x = self.relu(x)
            x += self.linear(emb).reshape(1, -1, 1, 1)

        #x = self.norm2(x)
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

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

"""
total params for 128 dim: 
142872835

- fix temb
- param count right?
"""

class UNet(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        channel_mult = (1, 2, 4, 8),
        ch=4,
        num_res_blocks=2,
        resolution=64,
        use_timestep=False,
        ):

        super(UNet, self).__init__()
     
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.ch = channels
        self.temb_ch = self.ch*4
        num_resolutions = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.use_timestep = use_timestep

        channel_scale = [self.ch * i for i in channel_mult]

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.in_conv = nn.Conv2d(3, self.ch, kernel_size=7, padding="same")
        prev_dim = None

        # down 
        for scale in range(len(channel_mult)):
            # bloc
            # attn
            # add block and attn, combine into module
            dim = channel_scale[scale]

            for i in range(num_res_blocks):
                in_dim = dim
                if prev_dim is not None:
                    if prev_dim != dim:
                        in_dim = prev_dim

                res1 = ResnetBlock(in_dim, dim) 
                res2 = ResnetBlock(dim, dim)
                attn = AttnBlock(dim)

                down_block = torch.nn.Sequential(
                    res1,
                    res2,
                    attn
                )

                self.down.append(down_block)

                prev_dim = dim

            if scale != num_resolutions - 1:
                self.down.append(torch.nn.MaxPool2d(2, 2))

        self.mid = torch.nn.Sequential(
            ResnetBlock(prev_dim, prev_dim * 4),
            AttnBlock(prev_dim * 4),
            ResnetBlock(prev_dim * 4, prev_dim),
        )

        # up part of unet
        prev_dim=prev_dim * 2
        inner_dim = None
        for scale in range(len(channel_mult) - 1, -1, -1):

            # blocc
            # attn
            # add block and attn, combine into module

            dim = channel_scale[scale]

            for i in range(num_res_blocks):

                in_dim = dim
                if prev_dim is not None:
                    inner_dim = prev_dim
                    if prev_dim != dim:
                        in_dim = prev_dim
                
                res1 = ResnetBlock(in_dim, dim) 
                res2 = ResnetBlock(dim, dim)
                attn = AttnBlock(dim)

                down_block = torch.nn.Sequential(
                    res1,
                    res2,
                    attn
                )

                self.up.append(down_block)
                prev_dim = dim
           
            if scale != 0: 
                #upconv
                self.up.append(torch.nn.ConvTranspose2d(dim, dim // 2, 3, stride=2, padding=1, output_padding=1))

        #self.out_conv = nn.Conv2d(ch * 4, out_ch)
    def forward(self, x):

        x = self.in_conv(x)
        state = []
        for i, module in enumerate(self.down):
            if type(module) != nn.MaxPool2d:
                res1, res2, attn = module
                x = res1(x)
                x = res2(x)
                x = attn(x)
            else:
                # if this layer downsamples, we will add x to state
                state.append(x)
                x = module(x)
        
        state.append(x)

        x = self.mid(x)
        add_l = False

        x = torch.cat((x, state.pop()), 1)

        for module in self.up:
            if type(module) != nn.ConvTranspose2d:
                res1, res2, attn = module
                x = res1(x)
                x = res2(x)
                x = attn(x)
            else:
                # if this layer upsamples, we will pop off of state
                x = module(x)
                x = torch.cat((x, state.pop()), 1)
        
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
    print(t)
    return torch.prod(1 - betas[:t])

def linear_beta_schedule(num_timesteps):
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)

def fake_normalize(t: torch.Tensor):
    return (t - torch.mean(t)) / torch.var(t)

class DiffusionModel():
    def __init__(self, eps_model, epochs=10, train_steps_per_epoch=-1):
        self.epochs = epochs
        self.train_steps = train_steps_per_epoch # -1 just means don't interrupt
        self.cum_loss = []
        self.eps_model = eps_model
        
        if beta_schedule is not None:
            self.beta_schedule = beta_schedule
        else:
            self.beta_schedule = linear_beta_schedule
    
    def log(self, loss):
        self.cum_loss.append(n_iter, n_epoch, loss.numel())
        
    # TRAINING LOOP
    def _train(dataset, eps_model, epochs=10, train_steps_per_epoch=-1):    
        img_shape = (1, 3, 32, 32)
        epochs = 10
        betas = linear_beta_schedule(n_noise_steps)
        optim = torch.optim.Adam(eps_model.parameters()) # we leave default params
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            for i, img in enumerate(dataset):

                if train_steps_per_epoch != -1:
                    if i == train_steps_per_epoch:
                        return 

                t = torch.randint(0, NUM_STEPS, (1,))[0]
                x_0 = img
                alpha = calculate_alpha(betas, t)
                eps = torch.normal(torch.zeros(img_shape), 1)
                loss = loss_fn(eps, eps_model(x_0, t)) 
                
                loss.backward()
                optim.step()

    def _sample(self, eps_model, num_steps=50):
        img_shape = (1, 3, 32, 32)
        betas = linear_beta_schedule(n_noise_steps)
        x = torch.normal(torch.zeros(img_shape), 1)
        for t in range(0, num_steps, -1):
            sig = alpha ** 2
            x = (1 / alpha) * (x - ((1 - alpha) / ((1 - alpha) ** 2)) * eps_model(x, t))

        return x

if __name__ == "__main__":

    test_res()
    test_res_temb()
    test_sin()
    test_attn()
    test_unet()


    """
    ds = torchvision.datasets.CIFAR10(download=True, root=".")
    diffusion_ds = remove_labels(ds)
    diffusion_ds = totensor_ds(diffusion_ds)
    eps_model = UNet()
    diffusion = DiffusionModel()
    diffusion._sample(eps_model)
    diffusion._train(diffusion_ds, eps_model, epochs=1, train_steps_per_epoch=10)

    """
    # stuff for ipynb

    """
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])

    n_noise_steps = 50
    im = t(ds[0][0])

    betas = linear_beta_schedule(n_noise_steps)

    im_15 = noise_alpha(im, betas=betas, t=5)
    im_30 = noise_alpha(im, betas=betas, t=7)
    im_45 = noise_alpha(im, betas=betas, t=30)
    im_50 = noise_alpha(im, betas=betas, t=45)

    f, arr = plt.subplots(2, 2, figsize=(8, 8))
    arr[0,0].imshow(im_15.permute(1, 2, 0))
    arr[0,1].imshow(im_30.permute(1, 2, 0))
    arr[1,0].imshow(im_45.permute(1, 2, 0))
    arr[1,1].imshow(im_50.permute(1, 2, 0))
    """