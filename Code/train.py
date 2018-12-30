import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor, RandomWarp

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epochs",type=int,dest="epochs",help="number of epochs",default=100)
    parser.add_argument("-lr",type=float,dest="lr",help="learning rate",default=0.001)
    parser.add_argument("-d","--decay",type=float,dest="decay",help="weight decay",default=0.005)
    parser.add_argument("-m","--momentum",type=float,dest="momentum",help="learning momentum",default=0.9)

    args = parser.parse_args()
    train(args.epochs, args.lr, args.momentum, args.decay)

def train(epochs, lr, momentum, decay):
    dataset = Segmentation(transform = Compose([ \
      Pad(120, mode='symmetric'), \
      #RandomAffine((0, 90), (31, 31)), \
			#RandomFlip(), \
			#RandomWarp(),
			CenterCrop(572, 388), \
			ToTensor()
    ]))
    model = UNet(n_class = 2).cuda() if torch.cuda.is_available() else UNet(n_class = 2)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = decay)
    loss_log = []
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

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
            loss = criterion(outputs, label)
            loss.backward()
            
            epoch_loss = epoch_loss + loss.item()
            
            optimizer.step()

            if i % 10 == 0 :
                print("Epoch #{} Batch #{} Loss: {}".format(epoch,i,loss.item()))
        loss_log.append(epoch_loss)
        
        #print("Epoch",epoch," finished. Loss :",loss.item())
        epoch_loss = 0
    print(loss_log)

get_options()
