import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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
			'image': torchvision.transforms.ToTensor()(self.images),
			'segmented': torchvision.transforms.ToTensor()(self.annotations)
		}
        
		if torch.cuda.is_available():
			sample['image'] = sample['image'].cuda()
			sample['segmented'] = sample['segmented'].cuda()
        
		if self.transform:
			sample = self.transform(sample)

		return sample
	
	def image(self):
		return self.images

