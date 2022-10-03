import math
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ds = torchvision.datasets.CIFAR10(download=True, root=".")
diffusion_ds = remove_labels(ds)
diffusion_ds = totensor_ds(diffusion_ds)