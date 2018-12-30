import torch.nn as nn
from torch.nn.init import normal_ as N
import os.path as osp
import torch

# TODO : BatchNorm
# https://github.com/milesial/Pytorch-UNet/

class UNet(nn.Module):
    def __init__(self, n_class=1, bilinear=False, pad=0):
        super(UNet,self).__init__()
        
        #DOWNSAMPLING
        #conv1
        self.dconv1_1 = nn.Conv2d(1,64,3,padding=pad)
        self.drelu1_1 = nn.LeakyReLU()
        self.dconv1_2 = nn.Conv2d(64,64,3,padding=pad)
        self.drelu1_2 = nn.LeakyReLU()
        self.dpool1 = nn.MaxPool2d(2,stride=2)  #1/2
        
        #conv2
        self.dconv2_1 = nn.Conv2d(64,128,3,padding=pad)
        self.drelu2_1 = nn.LeakyReLU()
        self.dconv2_2 = nn.Conv2d(128,128,3,padding=pad)
        self.drelu2_2 = nn.LeakyReLU()
        self.dpool2 = nn.MaxPool2d(2,stride=2)  #1/4
        
        #conv3
        self.dconv3_1 = nn.Conv2d(128,256,3,padding=pad)
        self.drelu3_1 = nn.LeakyReLU()
        self.dconv3_2 = nn.Conv2d(256,256,3,padding=pad)
        self.drelu3_2 = nn.LeakyReLU()
        self.dpool3 = nn.MaxPool2d(2,stride=2)  #1/8
        self.ddrop3 = nn.Dropout2d(p=0.2)
        
        #conv4
        self.dconv4_1 = nn.Conv2d(256,512,3,padding=pad)
        self.drelu4_1 = nn.LeakyReLU()
        self.dconv4_2 = nn.Conv2d(512,512,3,padding=pad)
        self.drelu4_2 = nn.LeakyReLU()
        self.dpool4 = nn.MaxPool2d(2,stride=2)  #1/8
        self.ddrop4 = nn.Dropout2d(p=0.2)
        
        #BOTTLENECK
        self.bconv1_1 = nn.Conv2d(512,1024,3,padding=pad)
        self.brelu1_1 = nn.LeakyReLU()
        self.bconv1_2 = nn.Conv2d(1024,1024,3,padding=pad)
        self.brelu1_2 = nn.LeakyReLU()
        
        #UPSAMPLING
        #conv1
        if bilinear:
            self.upool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool1 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.uconv1_1 = nn.Conv2d(1024,512,3,padding=pad)
        self.urelu1_1 = nn.LeakyReLU()
        self.uconv1_2 = nn.Conv2d(512,512,3,padding=pad)
        self.urelu1_2 = nn.LeakyReLU()
        self.udrop1 = nn.Dropout2d(p=0.2)
        
        #conv2
        if bilinear:
            self.upool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool2 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.uconv2_1 = nn.Conv2d(512,256,3,padding=pad)
        self.urelu2_1 = nn.LeakyReLU()
        self.uconv2_2 = nn.Conv2d(256,256,3,padding=pad)
        self.urelu2_2 = nn.LeakyReLU()
        self.udrop2 = nn.Dropout2d(p=0.2)
        
        #conv3
        if bilinear:
            self.upool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool3 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.uconv3_1 = nn.Conv2d(256,128,3,padding=pad)
        self.urelu3_1 = nn.LeakyReLU()
        self.uconv3_2 = nn.Conv2d(128,128,3,padding=pad)
        self.urelu3_2 = nn.LeakyReLU()
        self.udrop3 = nn.Dropout2d(p=0.2)
        
        #conv4
        if bilinear:
            self.upool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool4 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.uconv4_1 = nn.Conv2d(128,64,3,padding=pad)
        self.urelu4_1 = nn.LeakyReLU()
        self.uconv4_2 = nn.Conv2d(64,64,3,padding=pad)
        self.urelu4_2 = nn.LeakyReLU()
        self.udrop4 = nn.Dropout2d(p=0.2)
        
        self.seg = nn.Conv2d(64,n_class,1,padding=pad)
        
        self._init_weights()
        #self._initialize_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                #nn.init.xavier_uniform(conv1.weight)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            #if isinstance(m, nn.ConvTranspose2d):
                #assert m.kernel_size[0] == m.kernel_size[1]
                #initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                #m.weight.data.copy_(initial_weight)

    def concat(self, x2, x1):
        #x1 has bigger dimensions than x2
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x1 = nn.functional.pad(x1,(-diffX//2,-diffX+diffX//2,-diffY//2,-diffY+diffY//2))
        x2 = torch.cat([x2,x1],dim=1)
        return x2
    
    def forward(self, x):
        h = x
        #DOWNSAMPLING
        return nn.Conv2d(1,2,3,padding=0)(h)
        h = self.drelu1_1(self.dconv1_1(h))
        h = self.drelu1_2(self.dconv1_2(h))
        cc1 = h
        h = self.dpool1(h)
        
        h = self.drelu2_1(self.dconv2_1(h))
        h = self.drelu2_2(self.dconv2_2(h))
        cc2 = h
        h = self.dpool2(h)
        
        h = self.drelu3_1(self.dconv3_1(h))
        h = self.drelu3_2(self.dconv3_2(h))
        cc3 = h
        #h = self.ddrop3(h)
        h = self.dpool3(h)
        
        h = self.drelu4_1(self.dconv4_1(h))
        h = self.drelu4_2(self.dconv4_2(h))
        cc4 = h
        #h = self.ddrop4(h)
        h = self.dpool4(h)
        
        #BOTTLENECK
        h = self.brelu1_1(self.bconv1_1(h))
        h = self.brelu1_2(self.bconv1_2(h))
        
        #UPSAMPLING
        h = self.upool1(h)
        h = self.concat(h,cc4)
        #h = self.udrop1(h)
        h = self.urelu1_1(self.uconv1_1(h))
        h = self.urelu1_2(self.uconv1_2(h))
        
        h = self.upool2(h)
        h = self.concat(h,cc3)
        #h = self.udrop2(h)
        h = self.urelu2_1(self.uconv2_1(h))
        h = self.urelu2_2(self.uconv2_2(h))
        
        h = self.upool3(h)
        h = self.concat(h,cc2)
        #h = self.udrop3(h)
        h = self.urelu3_1(self.uconv3_1(h))
        h = self.urelu3_2(self.uconv3_2(h))
        
        h = self.upool4(h)
        h = self.concat(h,cc1)
        #h = self.udrop4(h)
        h = self.urelu4_1(self.uconv4_1(h))
        h = self.urelu4_2(self.uconv4_2(h))
        
        h = self.seg(h)
        
        return h
