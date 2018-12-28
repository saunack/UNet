import torch.nn as nn
import os.path as osp
import torch

#TODO : BatchNorm
# https://github.com/milesial/Pytorch-UNet/

class UNet(nn.Module):
    def __init__(self,n_class = 1,bilinear = False,pad=1):
        super(UNet,self).__init__()
        
        self.drop = nn.Dropout2d(p=0.2)
        #DOWNSAMPLING
        #conv1
        self.dconv1_1 = nn.Conv2d(1,64,3,padding=pad)
        self.drelu1_1 = nn.ReLU(inplace=True)
        self.dconv1_2 = nn.Conv2d(64,64,3,padding=pad)
        self.drelu1_2 = nn.ReLU(inplace=True)
        self.dpool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)  #1/2
        
        #conv2
        self.dconv2_1 = nn.Conv2d(64,128,3,padding=pad)
        self.drelu2_1 = nn.ReLU(inplace=True)
        self.dconv2_2 = nn.Conv2d(128,128,3,padding=pad)
        self.drelu2_2 = nn.ReLU(inplace=True)
        self.dpool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)  #1/4
        
        #conv3
        self.dconv3_1 = nn.Conv2d(128,256,3,padding=pad)
        self.drelu3_1 = nn.ReLU(inplace=True)
        self.dconv3_2 = nn.Conv2d(256,256,3,padding=pad)
        self.drelu3_2 = nn.ReLU(inplace=True)
        self.dpool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)  #1/8
        
        #conv4
        self.dconv4_1 = nn.Conv2d(256,512,3,padding=pad)
        self.drelu4_1 = nn.ReLU(inplace=True)
        self.dconv4_2 = nn.Conv2d(512,512,3,padding=pad)
        self.drelu4_2 = nn.ReLU(inplace=True)
        self.dpool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)  #1/8
        
        #BOTTLENECK
        self.bconv1_1 = nn.Conv2d(512,1024,3,padding=pad)
        self.brelu1_1 = nn.ReLU(inplace=True)
        self.bconv1_2 = nn.Conv2d(1024,1024,3,padding=pad)
        self.brelu1_2 = nn.ReLU(inplace=True)
        
        #UPSAMPLING
        #conv1
        if bilinear:
            self.upool1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool1 = nn.ConvTranspose2d(1024,1024,3,stride=2,bias=False)
        self.uconv1_1 = nn.Conv2d(1024+512,512,3,padding=pad)
        self.urelu1_1 = nn.ReLU(inplace=True)
        self.uconv1_2 = nn.Conv2d(512,512,3,padding=pad)
        self.urelu1_2 = nn.ReLU(inplace=True)
        
        #conv2
        if bilinear:
            self.upool2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool2 = nn.ConvTranspose2d(512,512,3,stride=2,bias=False)
        self.uconv2_1 = nn.Conv2d(512+256,256,3,padding=pad)
        self.urelu2_1 = nn.ReLU(inplace=True)
        self.uconv2_2 = nn.Conv2d(256,256,3,padding=pad)
        self.urelu2_2 = nn.ReLU(inplace=True)
        
        #conv3
        if bilinear:
            self.upool3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool3 = nn.ConvTranspose2d(256,256,3,stride=2,bias=False)
        self.uconv3_1 = nn.Conv2d(256+128,128,3,padding=pad)
        self.urelu3_1 = nn.ReLU(inplace=True)
        self.uconv3_2 = nn.Conv2d(128,128,3,padding=pad)
        self.urelu3_2 = nn.ReLU(inplace=True)
        
        #conv4
        if bilinear:
            self.upool4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upool4 = nn.ConvTranspose2d(128,128,3,stride=2,bias=False)
        self.uconv4_1 = nn.Conv2d(128+64,64,3,padding=pad)
        self.urelu4_1 = nn.ReLU(inplace=True)
        self.uconv4_2 = nn.Conv2d(64,64,3,padding=pad)
        self.urelu4_2 = nn.ReLU(inplace=True)
        
        self.seg = nn.Conv2d(64,n_class,1,padding=pad)
        
#        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def concat(self, x2, x1):
        #x1 has bigger dimensions than x2
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = nn.functional.pad(x2,(diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2))
        x2 = torch.cat([x2,x1],dim=1)
        return x2
    
    def forward(self, x):
        h = x
        #DOWNSAMPLING
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
        h = self.drop(h)
        h = self.dpool3(h)
        
        h = self.drelu4_1(self.dconv4_1(h))
        h = self.drelu4_2(self.dconv4_2(h))
        cc4 = h
        h = self.drop(h)
        h = self.dpool4(h)
        
        #BOTTLENECK
        h = self.brelu1_1(self.bconv1_1(h))
        h = self.brelu1_2(self.bconv1_2(h))
        
        #UPSAMPLING
        h = self.upool1(h)
        h = self.concat(h,cc4)
        h = self.drop(h)
        h = self.urelu1_1(self.uconv1_1(h))
        h = self.urelu1_2(self.uconv1_2(h))
        
        h = self.upool2(h)
        h = self.concat(h,cc3)
        h = self.drop(h)
        h = self.urelu2_1(self.uconv2_1(h))
        h = self.urelu2_2(self.uconv2_2(h))
        
        h = self.upool3(h)
        h = self.concat(h,cc2)
        h = self.drop(h)
        h = self.urelu3_1(self.uconv3_1(h))
        h = self.urelu3_2(self.uconv3_2(h))
        
        h = self.upool4(h)
        h = self.concat(h,cc1)
        h = self.drop(h)
        h = self.urelu4_1(self.uconv4_1(h))
        h = self.urelu4_2(self.uconv4_2(h))
        
        h = self.seg(h)
        
        return h
