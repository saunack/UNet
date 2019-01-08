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
model = UNet(n_class = 2).cuda() if torch.cuda.is_available() else UNet(n_class = 2)

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
def train(epochs=10, lr=0.001, display=False, save=False, load=False):
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    loss_log = []

    if load:
        get_checkpoint(model, optimizer, loss_log)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(epochs):
        #print("Starting Epoch #{}".format(epoch))

        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        epoch_loss = 0

        for i,images in enumerate(train_loader):
            # get the inputs
            image, label = images['image'], images['label']
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            ## Run the forward pass
            outputs = model.forward(image).cuda() if torch.cuda.is_available() else model.forward(image)
         
            loss = criterion(outputs.float(), label)
            loss.backward()
            
            epoch_loss = epoch_loss + loss.item()
            
            optimizer.step()

            if i % 10 == 0 :
                print("Epoch #{} Batch #{} Loss: {}".format(epoch,i,loss.item()))
        loss_log.append(epoch_loss)
        
        #print("Epoch",epoch," finished. Loss :",loss.item())
        print(epoch,loss.item())
        epoch_loss = 0
    if save:
        save_checkpoint({'state_dict':model.state_dict(),
				                  'optimizer':optimizer.state_dict(),
													'loss_log':loss_log,
													},"unet.pth")
    print(loss_log)
    #T.ToPILImage()(outputs[0].float()).show()

    if display:
      testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
      dataiter = iter(testloader)

      testimg = dataiter.next()
      img, lbl = testimg['image'], testimg['label']
      trained = model(img)
      T.ToPILImage()(img[0]).show()
      T.ToPILImage()(lbl.float()).show()
      T.ToPILImage()((trained[0][0]).unsqueeze(0).float()).show()
      T.ToPILImage()((trained[0][1]).unsqueeze(0).float()).show()
      T.ToPILImage()((torch.argmax(trained[0],dim=0)).unsqueeze(0).float()).show()

      #matching = (thresholded[0].long() == lbl.long()).sum()
      #accuracy = float(matching) / lbl.numel()
      #print("matching {}, total {}, accuracy {}".format(matching, lbl.numel(),\
      #  accuracy))
