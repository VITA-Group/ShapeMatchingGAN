import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
import torch.nn.functional as F
import random
from utils import load_image, to_data, to_var

#######################  Texture Network
class VGGFeature(nn.Module):
    def __init__(self, cnn, gpu=True):
        super(VGGFeature, self).__init__()
        
        self.model1 = cnn[:2]
        self.model2 = cnn[2:7]
        self.model3 = cnn[7:12]
        self.model4 = cnn[12:21]
        self.model5 = cnn[21:30]
        cnn_normalization_mean = (torch.tensor([0.485, 0.456, 0.406]) + 1) * 0.5
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]) * 0.5
        self.cnn_normalization_mean = cnn_normalization_mean.view(-1, 1, 1)
        self.cnn_normalization_std = cnn_normalization_std.view(-1, 1, 1)
        if gpu:
            self.cnn_normalization_mean = self.cnn_normalization_mean.cuda()
            self.cnn_normalization_std = self.cnn_normalization_std.cuda()
        
    def forward(self, x):
        x = (x - self.cnn_normalization_mean) / self.cnn_normalization_std
        conv1_1 = self.model1(x)
        conv2_1 = self.model2(conv1_1)
        conv3_1 = self.model3(conv2_1)
        conv4_1 = self.model4(conv3_1)
        conv5_1 = self.model5(conv4_1)
        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
    
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

class SemanticFeature(nn.Module):
    def __init__(self):
        super(SemanticFeature, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        conv1_1 = x
        conv2_1 = self.pool1(conv1_1)
        conv3_1 = self.pool2(conv2_1)
        conv4_1 = self.pool3(conv3_1)
        conv5_1 = self.pool4(conv4_1)
        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]    
    
def get_GRAM(filename, VGGFeatures, batchsize, gpu=True):
    getmask = SemanticFeature()
    for param in getmask.parameters():
        param.requires_grad = False
    
    img = load_image(filename)
    target = img[:,:,:,img.size(3)//2:img.size(3)]
    target_mask = img[:,:,:,0:img.size(3)//2]
    if gpu:
        getmask.cuda()
        target = to_var(target)
        target_mask = to_var(target_mask)

    with torch.no_grad():
        tmaps_fore = [(A.detach()+1)*0.5 for A in getmask(target_mask[:,0:1])]
        tmaps_back = [1-A for A in tmaps_fore]
        style_targets1 = [GramMatrix()(A*tmaps_fore[a]).detach() for a, A in enumerate(VGGFeatures(target))]
        style_targets2 = [GramMatrix()(A*tmaps_back[a]).detach() for a, A in enumerate(VGGFeatures(target))]

    for i in range(len(style_targets1)):
        style_targets1[i] = style_targets1[i].repeat(batchsize,1,1)
        style_targets2[i] = style_targets2[i].repeat(batchsize,1,1)    
        
    return [style_targets1, style_targets2]
