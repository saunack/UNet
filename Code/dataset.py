import torch
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from deform import elastic
import numpy as np

class ToTensor(object):
	""" Convert PIL Image to Tensor """

	def __call__(self, sample):
		img = F.to_tensor(sample['image'])
		seg = F.to_tensor(sample['segmented'])

		return {'image': img, 'segmented': seg}

class RandomWarp(object):
	""" Randomly apply elastic deformation to PIL Image """

	"""
	Args:
		p (float, optional): probability of warping
		alpha (float, optional): mean of gaussian
		sigma (float, optional): stddev of gaussian
	"""

	def __init__(self, p=0.5, alpha=40, sigma=2000):
		self.p = 2.0
		self.param, self.deform = elastic(alpha, sigma)

	def __call__(self, sample):
		img, seg = sample['image'], sample['segmented']
		
		if random.random() < self.p:
			dx, dy = self.param(np.array(img).size)
			img = self.deform(img, dx, dy)
			seg = self.deform(seg, dx, dy)

		return {'image': img, 'segmented': seg}

class CenterCrop(object):
	""" Crop given image in sample """

	"""
	Args:
		size (int): size of cropped image
	"""

	def __init__(self, img_size, seg_size):
		self.img_size = img_size
		self.seg_size = seg_size
	
	def __call__(self, sample):
		img = F.center_crop(sample['image'], self.img_size)
		seg = F.center_crop(sample['segmented'], self.seg_size)

		return {'image': img, 'segmented': seg}

class RandomFlip(object):
	""" Randomly flip given image in sample """

	"""
	Args:
		p (tuple or int): probability of flipping horizontally and vertically
	"""

	def __init__(self, p=0.5):
		if isinstance(p, list):
			self.p = p
		else:
			self.p = (p, p)
	
	def __call__(self, sample):
		img, seg = sample['image'], sample['segmented']

		if random.random() < self.p[0]:
			# print("Horizontal flip")
			img = F.hflip(img)
			seg = F.hflip(seg)

		if random.random() < self.p[1]:
			# print("Vertical flip")
			img = F.vflip(img)
			seg = F.vflip(seg)

		return {'image': img, 'segmented': seg}

class Pad(object):
	""" Pad given image in sample """

	"""
	Args:
		padding (int): Padding value
		mode (string, optional): allowed modes same as in 
			torchvision.transforms.Pad
	"""

	def __init__(self, padding, mode='constant'):
		self.padding = padding
		self.mode = mode
	
	def __call__(self, sample):
		img = F.pad(sample['image'], self.padding, padding_mode=self.mode)
		seg = F.pad(sample['segmented'], self.padding, padding_mode=self.mode)
		# print("Padding - size {}, mode {}".format(self.padding, self.mode))

		return {'image': img, 'segmented': seg}

class RandomAffine(object):
	""" Randomly rotate and translate the image in sample """

	"""
	Args:
		degrees (tuple): Range of rotation
		translate (tuple): maximum absolute horizontal and 
			vertical translations
	"""

	def __init__(self, degrees, translate):
		self.degrees = degrees
		self.translate = translate

	def __call__(self, sample):
		degrees, translate, _, _ = T.RandomAffine.get_params(self.degrees, self.translate,
			None, None, [1,1])
		# print("Affine - rotate {} degrees, translate ({}, {})".format(degrees, translate[0], translate[1]))
		
		img = F.affine(sample['image'], degrees, translate, 1, 0)
		seg = F.affine(sample['segmented'], degrees, translate, 1, 0)

		return {'image': img, 'segmented': seg}

class Segmentation(Dataset):
	""" Segmentation dataset """

	def __init__(self,transform = None):
		"""
		Args:
			annotations (string): Path to segmented labels (format 'tiff')
			images (string): Path to image files (format 'tiff')
			transform (callable, optional): Optional transform to be applied
				on a sample
		"""

		self.images = Image.open('../Data/train-volume.tif')
		self.annotations = Image.open('../Data/train-labels.tif')
		self.transform = transform
	
	def __len__(self):
		return 30
	
	def __getitem__(self, idx):
		self.images.seek(idx)
		self.annotations.seek(idx)

		#sample = (torchvision.transforms.ToTensor()(self.images)[0],
			#torchvision.transforms.ToTensor()(self.annotations)[0])

		sample = {
			'image': self.images,
			'segmented': self.annotations
		}
        
		if self.transform:
			sample = self.transform(sample)
		
		if torch.cuda.is_available():
			sample['image'] = sample['image'].cuda()
			sample['segmented'] = sample['segmented'].cuda()

		sample['segmented'] = sample['segmented'][0].long()

		return sample
	
	def get_images(self):
		return self.images
