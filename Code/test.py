import torch
from torchvision import transforms as T
from PIL import Image
from dataset import Segmentation, RandomFlip, Pad, RandomAffine, CenterCrop, ToTensor, RandomWarp
from model import UNet
from dd import deform_grid

t = T.Compose([
	#RandomFlip(), 
	Pad(150, mode='symmetric'),
	#RandomAffine((0, 90), (31, 31)),
	#RandomWarp(),
	CenterCrop(572, 388), 
	ToTensor()
])
transform = T.Compose([ \
      Pad(150, mode='symmetric'), \
      RandomAffine((0, 90), (31, 31)), \
			RandomFlip(), \
			RandomWarp(),
			CenterCrop(572, 388), \
			ToTensor()
    ])
 
dataset = Segmentation(transform=t)
original = Segmentation(transform=ToTensor())

pil = T.ToPILImage()

def seg(sample):
	pil(sample['segmented'].unsqueeze(0).float()).show()
	return sample['segmented'].unsqueeze(0)

def img(sample):
	pil(sample['image']).show()
	return sample['image']

model = UNet(n_class=2)

def show_warp(sigma=10, alpha=30, kernel_dim=23):
	I = Image.open('../Data/train-volume.tif')
	grid = deform_grid(img_size = I.size[0], sigma=sigma, kernel_dim=kernel_dim, alpha=alpha)
	O = pil(torch.nn.functional.grid_sample(T.ToTensor()(I).unsqueeze(0),grid)[0])
	I.show()
	O.show()
