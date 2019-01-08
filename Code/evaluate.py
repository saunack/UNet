import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Test_image, Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp

from torchvision import transforms as T
from PIL import Image
import numpy as np

# Dataset
dataset = Test_image(transform = Compose([ \
	Pad(150, mode='symmetric'), \
		CenterCrop(512, 512), \
			CenterCrop(512, 504), \
				ToTensor()
			]))

# Neural network
model = UNet(n_class = 1).cuda() if torch.cuda.is_available() else UNet(n_class = 1)

def get_checkpoint(model, optimizer, loss):
	filename = "unet.pth"
	map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	if os.path.isfile(filename):
		checkpoint = torch.load(filename, map_location=map_location)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		loss.append(checkpoint['loss_log'][0])
		
		#def train(epochs, lr, momentum, decay, display):
def evaluate():
	#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
	optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
	loss_log = []
	
	get_checkpoint(model, optimizer, loss_log)
	
	criterion = torch.nn.BCELoss(reduction='mean')
	
	testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
	dataiter = iter(testloader)
	testimg = dataiter.next()
	
	while True:
		try:
			testimg = dataiter.next()
			img= testimg['image']
			trained = model(img)
			thresholded = (trained > torch.tensor([0.5]))
			T.ToPILImage()(img[0]).show()
			T.ToPILImage()((trained[0]).float()).show()
			T.ToPILImage()((thresholded[0]).float()).show()
			
		
		except StopIteration:
			break
