import math
import torch
from torch import nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
NUM_STEPS = 50
SHAPE = (1, 3, 32, 32)
LATENT_SHAPE = (1, 2, 3)
def noise(x, beta):
    x = x * torch.normal(mean=sqrt_beta)
def generate_random_noise():
    return torch.normal(mean=0., std=torch.ones(SHAPE))
def create_betas():
    beta = 1
    betas = []
    for i in range(NUM_STEPS):
        betas.append(beta / NUM_STEPS)
    return betas
# reverse process:
class DDPM(nn.Module):
    def __init__(self):
        super(DDPM, self).__init__()
        self.mu = nn.Parameter(shape=LATENT_SHAPE)
        self.sigma = nn.Parameter(shape=LATENT_SHAPE)
    # forward process
    def _forward():
        x = generate_random_noise()
        betas = create_betas()
        for i in range(NUM_STEPS - 1):
            beta = betas[i]
            sqrt_beta = math.sqrt(1 - beta)
            x *= torch.normal(mean=sqrt_beta*x, std=beta, shape=SHAPE)
        return x
    def _reverse(x): 
        mu = torch.Parameter(shape=LATENT_SHAPE)
        sigma = torch.Parameter(shape=LATENT_SHAPE)
        for i in range(NUM_STEPS):
            x *= torch.normal(mean=mu, std=sigma, shape=SHAPE)
        return x
    def forward(x):
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.input_shape = (1, 3, 32, 32)
        self.output_shape = (1, 3, 32, 32)

        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
    
    def forward(self, x):
        print(x.shape)
        assert x.shape == self.input_shape
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        assert x.shape == self.output_shape

        """
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(channel_scales)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_timestep = use_timestep
        self.front_conv = torch.nn.Conv2d(self.in_channels, self.ch, 3, 1, 1)
        self.down = nn.ModuleList()
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
        """

class NoiseDataLoader(DataLoader):
    def __init__(self):
        super(NoiseDataLoader, self).__init__()
    def sample(self):
        pass

####### TESTS #######
def noise_test():
    betas = create_betas()
    for i in range(50):
        pass
def test_unet():
    u = UNet()
    x = generate_random_noise()
    x = u(x)

def noise(im):
    im = im.unsqueeze(0)
    t = [im]
    n = im
    betas = create_betas()
    for i in range(NUM_STEPS - 1):
        beta = betas[i]
        sqrt_beta = math.sqrt(beta)
        sqrt_beta * im
        im *= torch.normal(mean=sqrt_beta*im, std=torch.ones(SHAPE) * beta)
        t.append(im)
    return t
        
def make_diffusion_dataset():
    ds = torchvision.datasets.CIFAR10(root='.', download=True)
    im = []
    t = torchvision.transforms.ToTensor()
    for img, label in ds:
        im.append(img)
    for i, img in enumerate(im):
        noise_list = noise(t(img))
        im[i] = noise_list
    return im

if __name__ == "__main__":
    ds = make_diffusion_dataset()
    #test_unet()
    #model = DDPM()
    #x = torch.ones(LATENT_SHAPE)
        
