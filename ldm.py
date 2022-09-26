import torch
from torch import nn
import math
from torch.nn import functional as F

"""
Notes
Default UNet model (has attention in it which can then be used as "conditioning")
O: But do we really need attention here? What's wrong with adding

we make le custom unet
"""

def SWISH(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# something else idk
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

    def forward(self, x):
        
        x = F.relu(torch.nn.self.conv1(x))
        h = x
        x = F.relu(self.conv2(x))
        x = x + h

# take some time embeddings 
class ResnetBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        dropout=0.1,
        emb_channels=512
        ):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm_1 = Norm(in_channels)
        self.norm_2 = Norm(out_channels)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(drop)

        self.emb_proj = torch.nn.Linear(emb_channels, out_channels)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            else:
                self.resize_shortcut = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x, emb=None):
        x_prev = x
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x_prev = self.conv_shortcut(x_prev)
            else:

                x_prev = self.resize_shortcut(x_prev)

        if self.use_conv_shortcut():
            
        return x + h

"""
from stable-diffusion repo
"""

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

def create_downsample_block(in_channels, 
    out_channels, 
    emb_channels, 
    dropout):

    d = nn.Module()
    d.down = a
    d.attn = b

class UNet(nn.Module):
    def __init__(self,
        channels,
        out_channels,
        channel_scales=(1,2,4,8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True):

        super(UNet, self).__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(channel_scales)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        self.front_conv = torch.nn.Conv2d(self.in_channels, self.ch, 3, 1, 1)
        self.down = nn.ModuleList()
        """

        for scale in channel_scales:
            # block
            # attn
            # add block and attn, combine into module
            self.down.append(create_downsample_block(block_in, 
            block_out, 
            self.temb_ch, 
            dropout, 
            attn_type)
        w = 8
        assert w % 2 == 0
        """

        s = int(math.log(w))
        r = 56
        # 3 successive downsampling layers

        #input is (1, 3, 64, 64)

        self.downsample = torch.nn.MaxPool2d(2, 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
        
        # down part of unet
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        r *= 2
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(r // 2, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),

        )
        r *= 2
        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(r // 2, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU()

        )
        self.relu = torch.nn.ReLU()

        # mid part of unet

        #

        # up part of unet

        self.block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(r, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r // 2, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r // 2, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        r //= 2

        self.block_5 = torch.nn.Sequential(
            torch.nn.Conv2d(r, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r // 2, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r // 2, r // 2, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

        r //= 2

        self.block_6 = torch.nn.Sequential(
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r, r, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(r , 3, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x):

        x = self.block_1(x)
        x_1 = x
        x = self.downsample(x)
        x = self.block_2(x)
        x_2 = x
        x = self.downsample(x)
        x = self.block_3(x)
        x_3 = x

        x = self.block_4(x)
        x = self.upsample(x)
        x = self.block_5(x)
        x = self.upsample(x)
        x = self.block_6(x)

        return x

# k == 4, meaning we do 2 downsampling steps...
# if k == 8, we do 3 steps

class Encoder(nn.module):
    def __init__(self, num):

        model_list = []

        for i in range(num):
            res_block = ResBlock()

            if i != num - 1:
                up = torch.nn.Upsample(scale_factor=2, mode="nearest")
            else:
                up = torch.nn.Identity()
            
            model_list.append(res_block)
            model_list.append(down)
        
        self.model_list = torch.nn.Sequential(model_list)
    
    def forward(self, x):
        return self.model_list(x)

class Decoder(nn.module):
    def __init__(self, num):

        self.downsample = torch.nn.MaxPool2d(2, 2)
        model_list = []

        for i in range(num):
            res_block = ResBlock()

            if i != num - 1:
                down = Downsample()
            else:
                down = torch.nn.Identity()
            
            model_list.append(res_block)
            model_list.append(down)
        
        self.model_list = torch.nn.Sequential(model_list)
    
    def forward(self, x):
        return self.model_list(x)

# maybe we can just use someone else's autoencoder?
datasets = [
    'CelebA'
    'LSUNBedrooms'

]

# weights from https://github.com/Rayhane-mamah/Efficient-VDVAE
autoencoder_weights = {
    # same thing?
    'CelebA': 'https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebahq256_8bits_baseline_checkpoints.zip',
    'FFHQ': 'https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq256_8bits_baseline_checkpoints.zip'
}

def download_pretraied_autoencoder():
    pass

def main():
    x = torch.zeros(1, 3, 64, 64)
    u = UNet()
    print(u(x).shape)

if __name__ == "__main__":
    main()
