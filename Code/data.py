import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import *

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
		
		if self.transform:
			sample = self.transform(sample)

		return sample
	
	def image(self):
		return self.images

def train(epochs=2,pad = 2):
    dataset = Segmentation()
    model = UNet(n_class = 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.03, momentum = 0.8, weight_decay = 0.0005)
    loss_log = []
    criterion = nn.BCELoss()

    print("Starting training")

    for epoch in range(epochs):
        print("Starting Epoch #", epoch)

        train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
        epoch_loss = 0

        for i,images in enumerate(train_loader):
            # get the inputs
            image, label = images['image'], images['segmented']
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            ## Run the forward pass
            outputs = model.forward(image).cuda() 
            loss = criterion(outputs, label)
            loss.backward()
            
            loss_log.append(loss.item())
            
            epoch_loss = epoch_loss + loss.item()
            
            optimizer.step()

            print("Epoch #", epoch, "Batch #",i)
        
        print("Epoch ",epoch," finished. Loss : ",loss)
        epoch_loss = 0
        
train(1)
