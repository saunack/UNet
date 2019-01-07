import argparse, os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp

from torchvision import transforms as T
from PIL import Image
import numpy as np

dataset = Segmentation(transform = Compose([ \
  Pad(150, mode='symmetric'), \
  RandomAffine((0, 90), (30, 30)), \
	CenterCrop(512, 512), \
	RandomFlip(), \
	RandomWarp(),
	CenterCrop(512, 504), \
	ToTensor()
]))
model = UNet(n_class = 1).cuda() if torch.cuda.is_available() else UNet(n_class = 1)

save_model = False
load_model = False

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

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs",type=int,dest="epochs",help="number of epochs",default=100)
    parser.add_argument("-lr",type=float,dest="lr",help="learning rate",default=0.001)
    #parser.add_argument("-d","--decay",type=float,dest="decay",help="weight decay",default=0.005)
    #parser.add_argument("-m","--momentum",type=float,dest="momentum",help="learning momentum",default=0.9)
    parser.add_argument("--display", action = 'store_true')
    parser.add_argument("--save", action = 'store_true')
    parser.add_argument("--load", action = 'store_true')
    args = parser.parse_args()
    global save_model
    global load_model
    save_model = args.save
    load_model = args.load
    #train(args.epochs, args.lr, args.momentum, args.decay, args.display)
    train(args.epochs, args.lr, args.display)

#def train(epochs, lr, momentum, decay, display):
def train(epochs, lr, display):
    #optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    loss_log = []
    if load_model:
        get_checkpoint(model, optimizer, loss_log)
    criterion = torch.nn.BCELoss(reduction='mean')
    for epoch in range(epochs):
        #print("Starting Epoch #{}".format(epoch))

        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
        epoch_loss = 0

        for i,images in enumerate(train_loader):
            # get the inputs
            image, label = images['image'], images['segmented']
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            ## Run the forward pass
            outputs = model.forward(image).cuda() if torch.cuda.is_available() else model.forward(image)
         
            if display:
              T.ToPILImage()(outputs[0].float()).show()

            loss = criterion(outputs.float(), label.float().unsqueeze(0))
            loss.backward()
            
            epoch_loss = epoch_loss + loss.item()
            
            optimizer.step()

            if i % 10 == 0 :
                print("Epoch #{} Batch #{} Loss: {}".format(epoch,i,loss.item()))
        loss_log.append(epoch_loss)
        
        #print("Epoch",epoch," finished. Loss :",loss.item())
        print(epoch,loss.item())
        epoch_loss = 0
    if save_model:
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
      img, seg = testimg['image'], testimg['segmented']
      trained = model(img)
      thresholded = (trained > torch.tensor([0.5]))
      T.ToPILImage()(img[0]).show()
      T.ToPILImage()(seg.float()).show()
      T.ToPILImage()((trained[0]).float()).show()
      T.ToPILImage()((thresholded[0]).float()).show()

      matching = (thresholded[0].long() == seg.long()).sum()
      accuracy = float(matching) / seg.numel()
      print("matching {}, total {}, accuracy {}".format(matching, seg.numel(),\
        accuracy))

get_options()
