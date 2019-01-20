import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp

from torchvision import transforms as T
from PIL import Image
import numpy as np

# Dataset
dataset = Segmentation(transform = Compose([ \
  Pad(150, mode='symmetric'), \
  RandomAffine((0, 90), (30, 30)), \
	CenterCrop(512, 512), \
	RandomFlip(), \
	RandomWarp(),
	CenterCrop(512, 504), \
	ToTensor()
]))

# Neural network
model = UNet(n_class = 1).cuda() if torch.cuda.is_available() else UNet(n_class = 1)

def save_checkpoint(checkpt, filename):
	torch.save(checkpt,filename)

def get_checkpoint(model, optimizer, loss):
	filename = "unet.pth"
	map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	if os.path.isfile(filename):
		checkpoint = torch.load(filename, map_location=map_location)
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		loss.append(checkpoint['loss_log'][0])

#def train(epochs, lr, momentum, decay, display):
def validate(display=False):
	#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
	optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
	loss_log = []

	get_checkpoint(model, optimizer, loss_log)

	criterion = torch.nn.BCELoss(reduction='mean')


	testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
	dataiter = iter(testloader)

	i = 1
	
	while True:
		try:
			testimg = dataiter.next()
			img, lbl = testimg['image'], testimg['label']
			trained = model(img)
			thresholded = (trained > torch.tensor([0.6]))
			
			image = T.ToPILImage()(img[0])
			label = T.ToPILImage()(lbl.float())
			trained_img = T.ToPILImage()((trained[0]).float())
			threshold_img = T.ToPILImage()((thresholded[0]).float())

			directory = "../Results/"
			
			if not os.path.exists(directory):
				os.makedirs(directory)
			
			image.save(directory + str(i)+'_data.png',"PNG")
			label.save(directory + str(i)+'_label.png',"PNG")
			threshold_img.save(directory + str(i)+'_thresholded.png',"PNG")
			trained_img.save(directory + str(i)+'_training_output.png',"PNG")
			
			i = i + 1
			
			if display:
				image.show()
				label.show()
				trained_img.show()
				threshold_img.show()

			#if display:
				#T.ToPILImage()(img[0]).show()
				#T.ToPILImage()(lbl.float()).show()
				#T.ToPILImage()((trained[0]).float()).show()
				#T.ToPILImage()((thresholded[0]).float()).show()

			TP = ((thresholded[0].long() == lbl.long()) & (thresholded[0].long() == 1)).sum()
			TN = ((thresholded[0].long() == lbl.long()) & (thresholded[0].long() == 0)).sum()
			FP = ((thresholded[0].long() != lbl.long()) & (thresholded[0].long() == 1)).sum()
			FN = ((thresholded[0].long() != lbl.long()) & (thresholded[0].long() == 0)).sum()
			matching = (thresholded[0].long() == lbl.long()).sum()
			accuracy = float(matching) / lbl.numel()
#			print("matching {}, total {}, accuracy {}".format(matching, lbl.numel(), accuracy))
			try:
				precision = float(TP.float()/(TP.float()+FP.float()))
				recall = float(TP.float()/(TP.float()+FN.float()))
				F1 = float(2*precision*recall/(precision + recall))
				print("accuracy {}, precision {}, recall {}, F1 score{}".format(accuracy, precision, recall, F1))
			except FloatingPointError:
				continue
			except ZeroDivisionError:
				print(TP,TN,FP,FN)
				continue
		except StopIteration:
			break
