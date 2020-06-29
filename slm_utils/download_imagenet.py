import torch 
import torchvision 
import scipy

imnet =   torchvision.datasets.ImageNet('/userdata/smetzger/data/ImageNet/', split='train', download=True)