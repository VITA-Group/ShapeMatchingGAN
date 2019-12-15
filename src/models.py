import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
import torch.nn.functional as F
import random
from utils import gaussian, to_var, to_data, save_image
from vgg import GramMSELoss, SemanticFeature
import numpy as np
import math
import torch.autograd as autograd
from torch.autograd import Variable
import os

id = 0 # for saving network output to file during training

#######################  Texture Network
# based on Convolution-BatchNorm-ReLU
class myTConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(myTConv, self).__init__()
        
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=0, in_channels=in_channels)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))

class myTBlock(nn.Module):
    def __init__(self, num_filter=128, p=0.0):
        super(myTBlock, self).__init__()
        
        self.myconv = myTConv(num_filter=num_filter, stride=1, in_channels=128)
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=0, in_channels=128)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
        self.dropout = nn.Dropout(p=p)
        
    def forward(self, x):
        return self.dropout(self.relu(x+self.bn(self.conv(self.pad(self.myconv(x))))))

class TextureGenerator(nn.Module):
    def __init__(self, ngf = 32, n_layers = 5):
        super(TextureGenerator, self).__init__()
        
        modelList = []
        modelList.append(ReplicationPad2d(padding=4))
        modelList.append(Conv2d(out_channels=ngf, kernel_size=9, padding=0, in_channels=3))
        modelList.append(ReLU())
        modelList.append(myTConv(ngf*2, 2, ngf))
        modelList.append(myTConv(ngf*4, 2, ngf*2))
        
        for n in range(int(n_layers/2)): 
            modelList.append(myTBlock(ngf*4, p=0.0))
        # dropout to make model more robust
        modelList.append(myTBlock(ngf*4, p=0.5))
        for n in range(int(n_layers/2)+1,n_layers):
            modelList.append(myTBlock(ngf*4, p=0.0))  
        
        modelList.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4))
        modelList.append(BatchNorm2d(num_features=ngf*2, track_running_stats=True))
        modelList.append(ReLU())
        modelList.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2))
        modelList.append(BatchNorm2d(num_features=ngf, track_running_stats=True))
        modelList.append(ReLU())
        modelList.append(ReplicationPad2d(padding=1))
        modelList.append(Conv2d(out_channels=3, kernel_size=9, padding=0, in_channels=ngf))
        modelList.append(Tanh())
        self.model = nn.Sequential(*modelList)
        
    def forward(self, x):
        return self.model(x)

###################### Glyph Network    
# based on Convolution-BatchNorm-LeakyReLU    
class myGConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(myGConv, self).__init__()
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=0, in_channels=in_channels)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        # either ReLU or LeakyReLU is OK
        self.relu = LeakyReLU(0.2)
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))

class myGBlock(nn.Module):
    def __init__(self, num_filter=128):
        super(myGBlock, self).__init__()
        
        self.myconv = myGConv(num_filter=num_filter, stride=1, in_channels=num_filter)
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=0, in_channels=num_filter)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        
    def forward(self, x):
        return x+self.bn(self.conv(self.pad(self.myconv(x))))

# Controllable ResBlock
class myGCombineBlock(nn.Module):
    def __init__(self, num_filter=128, p=0.0):
        super(myGCombineBlock, self).__init__()
        
        self.myBlock1 = myGBlock(num_filter=num_filter)
        self.myBlock2 = myGBlock(num_filter=num_filter)
        self.relu = LeakyReLU(0.2)
        self.label = 1.0
        self.dropout = nn.Dropout(p=p)
        
    def myCopy(self):
        self.myBlock1.load_state_dict(self.myBlock2.state_dict())
        
    def forward(self, x):
        return self.dropout(self.relu(self.myBlock1(x)*self.label + self.myBlock2(x)*(1.0-self.label)))
    
class GlyphGenerator(nn.Module):
    def __init__(self, ngf=32, n_layers = 5):
        super(GlyphGenerator, self).__init__()
        
        encoder = []
        encoder.append(ReplicationPad2d(padding=4))
        encoder.append(Conv2d(out_channels=ngf, kernel_size=9, padding=0, in_channels=3))
        encoder.append(LeakyReLU(0.2))
        encoder.append(myGConv(ngf*2, 2, ngf))
        encoder.append(myGConv(ngf*4, 2, ngf*2))

        transformer = []
        for n in range(int(n_layers/2)-1):
            transformer.append(myGCombineBlock(ngf*4,p=0.0))
        # dropout to make model more robust    
        transformer.append(myGCombineBlock(ngf*4,p=0.5))
        transformer.append(myGCombineBlock(ngf*4,p=0.5))
        for n in range(int(n_layers/2)+1,n_layers):
            transformer.append(myGCombineBlock(ngf*4,p=0.0))  
        
        decoder = []
        decoder.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4))
        decoder.append(BatchNorm2d(num_features=ngf*2, track_running_stats=True))
        decoder.append(LeakyReLU(0.2))
        decoder.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2))
        decoder.append(BatchNorm2d(num_features=ngf, track_running_stats=True))
        decoder.append(LeakyReLU(0.2))
        decoder.append(ReplicationPad2d(padding=1))
        decoder.append(Conv2d(out_channels=3, kernel_size=9, padding=0, in_channels=ngf))
        decoder.append(Tanh())
        
        self.encoder = nn.Sequential(*encoder)
        self.transformer = nn.Sequential(*transformer)
        self.decoder = nn.Sequential(*decoder)
    
    def myCopy(self):
        for myCombineBlock in self.transformer:
            myCombineBlock.myCopy()
            
    # controlled by Controllable ResBlcok    
    def forward(self, x, l):
        for myCombineBlock in self.transformer:
            # label smoothing [-1,1]-->[0.9,0.1]
            myCombineBlock.label = (1.0-l)*0.4+0.1
        out0 = self.encoder(x)
        out1 = self.transformer(out0)
        out2 = self.decoder(out1)
        return out2

    
##################### Sketch Module 
# based on Convolution-InstanceNorm-ReLU   
# Smoothness Block
class myBlur(nn.Module):
    def __init__(self, kernel_size=121, channels=3):
        super(myBlur, self).__init__()
        kernel_size = int(int(kernel_size/2)*2)+1
        self.kernel_size=kernel_size
        self.channels = channels
        self.GF = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)
        x_cord = torch.arange(self.kernel_size+0.)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        self.xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        self.mean = (self.kernel_size - 1)//2
        self.diff = -torch.sum((self.xy_grid - self.mean)**2., dim=-1)
        self.gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                    kernel_size=self.kernel_size, groups=self.channels, bias=False)

        self.gaussian_filter.weight.requires_grad = False
        
    def forward(self, x, sigma, gpu):
        sigma = sigma * 8. + 16.
        variance = sigma**2.
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(self.diff /(2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)
        if gpu:
            gaussian_kernel = gaussian_kernel.cuda()
        self.gaussian_filter.weight.data = gaussian_kernel
        return self.gaussian_filter(F.pad(x, (self.mean,self.mean,self.mean,self.mean), "replicate")) 

class mySConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(mySConv, self).__init__()
        
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=1, in_channels=in_channels)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class mySBlock(nn.Module):
    def __init__(self, num_filter=128):
        super(mySBlock, self).__init__()
        
        self.myconv = mySConv(num_filter=num_filter, stride=1, in_channels=num_filter)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=1, in_channels=num_filter)
        self.bn = InstanceNorm2d(num_features=num_filter)
        self.relu = ReLU()
        
    def forward(self, x):
        return self.relu(x+self.bn(self.conv(self.myconv(x))))
    
# Transformation Block
class SketchGenerator(nn.Module):
    def __init__(self, in_channels = 4, ngf = 32, n_layers = 5):
        super(SketchGenerator, self).__init__()
        
        encoder = []
        encoder.append(Conv2d(out_channels=ngf, kernel_size=9, padding=4, in_channels=in_channels))
        encoder.append(ReLU())
        encoder.append(mySConv(ngf*2, 2, ngf))
        encoder.append(mySConv(ngf*4, 2, ngf*2))
        
        transformer = []
        for n in range(n_layers):
            transformer.append(mySBlock(ngf*4+1))
        
        decoder1 = []
        decoder2 = []
        decoder3 = []
        decoder1.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4+2))
        decoder1.append(InstanceNorm2d(num_features=ngf*2))
        decoder1.append(ReLU())
        decoder2.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2+1))
        decoder2.append(InstanceNorm2d(num_features=ngf))
        decoder2.append(ReLU())
        decoder3.append(Conv2d(out_channels=3, kernel_size=9, padding=1, in_channels=ngf+1))
        decoder3.append(Tanh())
        
        self.encoder = nn.Sequential(*encoder)
        self.transformer = nn.Sequential(*transformer)
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder2 = nn.Sequential(*decoder2)
        self.decoder3 = nn.Sequential(*decoder3)
    
    # controlled by label concatenation
    def forward(self, x, l):
        l_img = l.expand(l.size(0), l.size(1), x.size(2), x.size(3))
        out0 = self.encoder(torch.cat([x, l_img], 1))
        l_img0 = l.expand(l.size(0), l.size(1), out0.size(2), out0.size(3))
        out1 = self.transformer(torch.cat([out0, l_img0], 1))
        l_img1 = l.expand(l.size(0), l.size(1), out1.size(2), out1.size(3))
        out2 = self.decoder1(torch.cat([out1, l_img1], 1))
        l_img2 = l.expand(l.size(0), l.size(1), out2.size(2), out2.size(3))
        out3 = self.decoder2(torch.cat([out2, l_img2], 1))
        l_img3 = l.expand(l.size(0), l.size(1), out3.size(2), out3.size(3))
        out4 = self.decoder3(torch.cat([out3, l_img3], 1))
        return out4
    
    
################ Discriminators
# Glyph and Texture Networks: BN
# Sketch Module: IN, multilayer
class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=32, n_layers=3, multilayer=False, IN=False):
        super(Discriminator, self).__init__()
        
        modelList = []    
        outlist1 = []
        outlist2 = []
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=2, in_channels=in_channels))
        modelList.append(LeakyReLU(0.2))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            modelList.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=2, in_channels=ndf * nf_mult_prev))
            if IN:
                modelList.append(InstanceNorm2d(num_features=ndf * nf_mult))
            else:
                modelList.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
            modelList.append(LeakyReLU(0.2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        outlist1.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        
        outlist2.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        if IN:
            outlist2.append(InstanceNorm2d(num_features=ndf * nf_mult))
        else:
            outlist2.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
        outlist2.append(LeakyReLU(0.2))
        outlist2.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult))
        self.model = nn.Sequential(*modelList)
        self.out1 = nn.Sequential(*outlist1)
        self.out2 = nn.Sequential(*outlist2)
        self.multilayer = multilayer
        
    def forward(self, x):
        y = self.model(x)
        out2 = self.out2(y)
        if self.multilayer:
            out1 = self.out1(y)
            return torch.cat((out1.view(-1), out2.view(-1)), dim=0)
        else:
            return out2.view(-1)

        
######################## Sketch Module
class SketchModule(nn.Module):
    def __init__(self, G_layers = 6, D_layers = 5, ngf = 32, ndf = 32, gpu=True):
        super(SketchModule, self).__init__()
        
        self.G_layers = G_layers
        self.D_layers = D_layers
        self.ngf = ngf
        self.ndf = ndf
        self.gpu = gpu
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.lambda_adv = 1
        self.loss = nn.L1Loss()
        
        # Sketch Module = transformationBlock + smoothnessBlock
        # transformationBlock
        self.transBlock = SketchGenerator(4, self.ngf, self.G_layers)
        self.D_B = Discriminator(7, self.ndf, self.D_layers, True, True)
        # smoothnessBlock
        self.smoothBlock = myBlur()
        
        self.trainerG = torch.optim.Adam(self.transBlock.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD = torch.optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # FOR TESTING
    def forward(self, t, l):
        l = torch.tensor(l).float()
        tl = self.smoothBlock(t, l, self.gpu)
        label = l.repeat(1, 1, 1, 1)
        label = label.cuda() if self.gpu else label
        return self.transBlock(tl, label)
    
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        self.transBlock.apply(weights_init)
        self.D_B.apply(weights_init)
        
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def update_discriminator(self, t, l):
        label = torch.tensor(l).float()
        label = label.repeat(t.size(0), 1, 1, 1)
        label = to_var(label) if self.gpu else label
        real_label = label.expand(label.size(0), label.size(1), t.size(2), t.size(3))    
        with torch.no_grad():
            tl = self.smoothBlock(t, l, self.gpu)
            fake_text = self.transBlock(tl, label)
            # print(tl.size(), real_label.size(), fake_text.size())
            fake_concat = torch.cat((tl, real_label, fake_text), dim=1)
        fake_output = self.D_B(fake_concat)
        real_concat = torch.cat((tl, real_label, t), dim=1)
        real_output = self.D_B(real_concat)
        gp = self.calc_gradient_penalty(self.D_B, real_concat.data, fake_concat.data)
        LBadv = self.lambda_adv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD.zero_grad()
        LBadv.backward()
        self.trainerD.step()
        return (real_output.mean() - fake_output.mean()).data.mean() * self.lambda_adv
    
    def update_generator(self, t, l):
        label = torch.tensor(l).float()
        label = label.repeat(t.size(0), 1, 1, 1)
        label = label.cuda() if self.gpu else label
        real_label = label.expand(label.size(0), label.size(1), t.size(2), t.size(3)) 
        tl = self.smoothBlock(t, l, self.gpu)
        fake_text = self.transBlock(tl, label)
        fake_concat = torch.cat((tl, real_label, fake_text), dim=1)
        fake_output = self.D_B(fake_concat)
        LBadv = -fake_output.mean() * self.lambda_adv
        LBrec = self.loss(fake_text, t) * self.lambda_l1
        LB = LBadv + LBrec
        self.trainerG.zero_grad()
        LB.backward()
        self.trainerG.step()
        #global id
        #if id % 50 == 0:
        #    viz_img = to_data(torch.cat((t[0], tl[0], fake_text[0]), dim=2))
        #    save_image(viz_img, '../output/deblur_result%d.jpg'%id)
        #id += 1
        return LBadv.data.mean(), LBrec.data.mean()
    
    def one_pass(self, t, scales):
        l = random.choice(scales)
        LDadv = self.update_discriminator(t, l)
        LGadv, Lrec = self.update_generator(t, l)
        return [LDadv,LGadv,Lrec]
    
    
######################## ShapeMatchingGAN
class ShapeMatchingGAN(nn.Module):
    def __init__(self, GS_nlayers = 6, DS_nlayers = 5, GS_nf = 32, DS_nf = 32,
                 GT_nlayers = 6, DT_nlayers = 5, GT_nf = 32, DT_nf = 32, gpu=True):
        super(ShapeMatchingGAN, self).__init__()
        
        self.GS_nlayers = GS_nlayers
        self.DS_nlayers = DS_nlayers
        self.GS_nf = GS_nf
        self.DS_nf = DS_nf
        self.GT_nlayers = GT_nlayers
        self.DT_nlayers = DT_nlayers
        self.GT_nf = GT_nf
        self.DT_nf = DT_nf        
        self.gpu = gpu
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.lambda_sadv = 0.1
        self.lambda_gly = 1.0
        self.lambda_tadv = 1.0
        self.lambda_sty = 0.01
        self.style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        self.loss = nn.L1Loss()
        self.gramloss = GramMSELoss()
        self.gramloss = self.gramloss.cuda() if self.gpu else self.gramloss
        self.getmask = SemanticFeature()
        for param in self.getmask.parameters():
            param.requires_grad = False

        self.G_S = GlyphGenerator(self.GS_nf, self.GS_nlayers)
        self.D_S = Discriminator(3, self.DS_nf, self.DS_nlayers)
        self.G_T = TextureGenerator(self.GT_nf, self.GT_nlayers)
        self.D_T = Discriminator(6, self.DT_nf, self.DT_nlayers)
        
        self.trainerG_S = torch.optim.Adam(self.G_S.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_S = torch.optim.Adam(self.D_S.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerG_T = torch.optim.Adam(self.G_T.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD_T = torch.optim.Adam(self.D_T.parameters(), lr=0.0002, betas=(0.5, 0.999))    
    
    # FOR TESTING
    def forward(self, x, l):
        x[:,0:1] = gaussian(x[:,0:1], stddev=0.2)
        xl = self.G_S(x, l) 
        xl[:,0:1] = gaussian(xl[:,0:1], stddev=0.2)
        return self.G_T(xl)
            
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        self.G_S.apply(weights_init)
        self.D_S.apply(weights_init)
        self.G_T.apply(weights_init)
        self.D_T.apply(weights_init)
        
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def update_structure_discriminator(self, x, xl, l):   
        with torch.no_grad():
            fake_x = self.G_S(xl, l)
        fake_output = self.D_S(fake_x)
        real_output = self.D_S(x)
        gp = self.calc_gradient_penalty(self.D_S, x.data, fake_x.data)
        LSadv = self.lambda_sadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD_S.zero_grad()
        LSadv.backward()
        self.trainerD_S.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_sadv
    
    def update_structure_generator(self, x, xl, l, t=None):
        fake_x = self.G_S(xl, l)
        fake_output = self.D_S(fake_x)
        LSadv = -fake_output.mean()*self.lambda_sadv
        LSrec = self.loss(fake_x, x) * self.lambda_l1
        LS = LSadv + LSrec
        if t is not None:
            # weight map based on the distance field 
            # whose pixel value increases with its distance to the nearest text contour point of t
            Mt = (t[:,1:2]+t[:,2:3])*0.5+1.0
            t_noise = t.clone()
            t_noise[:,0:1] = gaussian(t_noise[:,0:1], stddev=0.2)
            fake_t = self.G_S(t_noise, l)
            LSgly = self.loss(fake_t*Mt, t*Mt) * self.lambda_gly
            LS = LS + LSgly
        self.trainerG_S.zero_grad()
        LS.backward()
        self.trainerG_S.step()
        #global id
        #if id % 60 == 0:
        #    viz_img = to_data(torch.cat((x[0], xl[0], fake_x[0]), dim=2))
        #    save_image(viz_img, '../output/structure_result%d.jpg'%id)
        #id += 1
        return LSadv.data.mean(), LSrec.data.mean(), LSgly.data.mean() if t is not None else 0
    
    def structure_one_pass(self, x, xl, l, t=None):
        LDadv = self.update_structure_discriminator(x, xl, l)
        LGadv, Lrec, Lgly = self.update_structure_generator(x, xl, l, t)
        return [LDadv, LGadv, Lrec, Lgly]    
    
    def update_texture_discriminator(self, x, y):
        with torch.no_grad():
            fake_y = self.G_T(x)          
            fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output = self.D_T(fake_concat)
        real_concat = torch.cat((x, y), dim=1)
        real_output = self.D_T(real_concat)
        gp = self.calc_gradient_penalty(self.D_T, real_concat.data, fake_concat.data)
        LTadv = self.lambda_tadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp)
        self.trainerD_T.zero_grad()
        LTadv.backward()
        self.trainerD_T.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_tadv        

    def update_texture_generator(self, x, y, t=None, l=None, VGGfeatures=None, style_targets=None):
        fake_y = self.G_T(x)
        fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output = self.D_T(fake_concat)
        LTadv = -fake_output.mean()*self.lambda_tadv
        Lrec = self.loss(fake_y, y) * self.lambda_l1
        LT = LTadv + Lrec
        if t is not None:
            with torch.no_grad():
                t[:,0:1] = gaussian(t[:,0:1], stddev=0.2)
                source_mask = self.G_S(t, l).detach()
                source = source_mask.clone()
                source[:,0:1] = gaussian(source[:,0:1], stddev=0.2)
                smaps_fore = [(A.detach()+1)*0.5 for A in self.getmask(source_mask[:,0:1])]
                smaps_back = [1-A for A in smaps_fore]
            fake_t = self.G_T(source)
            out = VGGfeatures(fake_t)
            style_losses1 = [self.style_weights[a] * self.gramloss(A*smaps_fore[a], style_targets[0][a]) for a,A in enumerate(out)]
            style_losses2 = [self.style_weights[a] * self.gramloss(A*smaps_back[a], style_targets[1][a]) for a,A in enumerate(out)]
            Lsty = (sum(style_losses1)+ sum(style_losses2)) * self.lambda_sty
            LT = LT + Lsty
        #global id
        #if id % 20 == 0:
        #    viz_img = to_data(torch.cat((x[0], y[0], fake_y[0]), dim=2))
        #    save_image(viz_img, '../output/texturee_result%d.jpg'%id)
        #id += 1             
        self.trainerG_T.zero_grad()
        LT.backward()
        self.trainerG_T.step()   
        return LTadv.data.mean(), Lrec.data.mean(), Lsty.data.mean() if t is not None else 0
    
    def texture_one_pass(self, x, y, t=None, l=None, VGGfeatures=None, style_targets=None):
        LDadv = self.update_texture_discriminator(x, y)
        LGadv, Lrec, Lsty = self.update_texture_generator(x, y, t, l, VGGfeatures, style_targets)
        return [LDadv, LGadv, Lrec, Lsty]
    
    def save_structure_model(self, filepath, filename):     
        torch.save(self.G_S.state_dict(), os.path.join(filepath, filename+'-GS.ckpt'))
        torch.save(self.D_S.state_dict(), os.path.join(filepath, filename+'-DS.ckpt'))
    def save_texture_model(self, filepath, filename):
        torch.save(self.G_T.state_dict(), os.path.join(filepath, filename+'-GT.ckpt'))
        torch.save(self.D_T.state_dict(), os.path.join(filepath, filename+'-DT.ckpt'))
