import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import Compose
from model import UNet
from dataset import Segmentation, RandomAffine, Pad, RandomFlip, CenterCrop, ToTensor

def train(epochs = 2):
    dataset = Segmentation(transform = Compose([ \
      Pad(120, mode='symmetric'), \
      RandomAffine((0, 90), (31, 31)), \
			RandomFlip(), \
			CenterCrop(572, 388), \
			ToTensor()
    ]))
    model = UNet(n_class = 2).cuda() if torch.cuda.is_available() else UNet(n_class = 2)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.03, momentum = 0.99, weight_decay = 0.0005)
    loss_log = []
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Starting Epoch #{}".format(epoch))

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
            
            loss_log.append(loss.item())
            
            epoch_loss = epoch_loss + loss.item()
            
            optimizer.step()

            print("Epoch #{} Batch #{} Loss: {}".format(epoch,i,loss.item()))
        
        print("Epoch",epoch," finished. Loss :",loss)
        epoch_loss = 0
        
train(6)
