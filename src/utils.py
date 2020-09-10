import torch

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import random
import os

pil2tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])])
tensor2pil = transforms.ToPILImage()

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    
def load_image(filename, load_type=0):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:
        img = Image.open(filename)
    else:
        img = text_image_preprocessing(filename)
    img = transform(img)
        
    return img.unsqueeze(dim=0)

def save_image(img, filename):
    tmp = ((img.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))

# black and white text image to distance-based text image
def text_image_preprocessing(filename):
    I = np.array(Image.open(filename))
    BW = I[:,:,0] > 127
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel>32]=32
    B_channel = pyimg.distance_transform_edt(1-BW)
    B_channel[B_channel>200]=200
    I[:,:,1] = G_channel.astype('uint8')
    I[:,:,2] = B_channel.astype('uint8')
    return Image.fromarray(I)

def gaussian(ins, mean = 0, stddev = 0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return torch.clamp(ins + noise, -1, 1)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('my') == -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# prepare batched filenames of all training data
# [[list of file names in one batch],[list of file names in one batch],...,[]]
def load_train_batchfnames(path, batch_size, usenum=3, trainnum=100000):
    trainnum = int((int(trainnum) // int(batch_size)) * batch_size)
    fnames = [('%04d.png' % (i%usenum)) for i in range(trainnum)]
    random.shuffle(fnames)
    trainbatches = [([]) for _ in range(trainnum//batch_size)]
    count = 0
    for i in range(trainnum//batch_size):
        traindatas = []
        for j in range(batch_size):
            traindatas += [os.path.join(path, fnames[count])]
            count = count + 1
        trainbatches[i] += traindatas
    return trainbatches

# prepare a batch of text images {t}
def prepare_text_batch(batchfnames, wd=256, ht=256, anglejitter=False):
    img_list = []
    for fname in batchfnames:
        img = Image.open(fname)     
        ori_wd, ori_ht = img.size
        w = random.randint(0,ori_wd-wd)
        h = random.randint(0,ori_ht-ht)
        img = img.crop((w,h,w+wd,h+ht))
        if anglejitter:
            random_angle = 90 * random.randint(0,3)
            img = img.rotate(random_angle)
        img = pil2tensor(img)         
        img = img.unsqueeze(dim=0)
        img_list.append(img)
    return torch.cat(img_list, dim=0)

# prepare {Xl, X, Y, fixed_noise} in PIL format from one image pair [X,Y]
def load_style_image_pair(filename, scales=[-1.0,-1./3,1./3,1.0], sketchmodule=None, gpu=True):
    img = Image.open(filename) 
    ori_wd, ori_ht = img.size
    ori_wd = ori_wd // 2
    X = pil2tensor(img.crop((0,0,ori_wd,ori_ht))).unsqueeze(dim=0)
    Y = pil2tensor(img.crop((ori_wd,0,ori_wd*2,ori_ht))).unsqueeze(dim=0)
    Xls = []    
    Noise = torch.tensor(0).float().repeat(1, 1, 1).expand(3, ori_ht, ori_wd)
    Noise = Noise.data.new(Noise.size()).normal_(0, 0.2)
    Noise = Noise.unsqueeze(dim=0)
    #Noise = tensor2pil((Noise+1)/2)    
    if sketchmodule is not None:
        X_ = to_var(X) if gpu else X
        for l in scales:  
            with torch.no_grad():
                Xl = sketchmodule(X_, l).detach()
            Xls.append(to_data(Xl) if gpu else Xl)            
    return [Xls, X, Y, Noise]


def rotate_tensor(x, angle):
    if angle == 1:
        return x.transpose(2, 3).flip(2)
    elif angle == 2:
        return x.flip(2).flip(3)
    elif angle == 3:
        return x.transpose(2, 3).flip(3)
    else:
        return x
    
# crop subimages for training 
# for structure transfer:  [Input,Output]=[Xl, X]
# for texture transfer:  [Input,Output]=[X, Y]
def cropping_training_batches(Input, Output, Noise, batchsize=16, anglejitter=False, wd=256, ht=256):
    img_list = []
    ori_wd = Input.size(2)
    ori_ht = Input.size(3)
    for i in range(batchsize):
        w = random.randint(0,ori_wd-wd)
        h = random.randint(0,ori_ht-ht)
        input = Input[:,:,w:w+wd,h:h+ht].clone()
        output = Output[:,:,w:w+wd,h:h+ht]
        noise = Noise[:,:,w:w+wd,h:h+ht]
        if anglejitter:
            random_angle = random.randint(0,3)
            input = rotate_tensor(input, random_angle)
            output = rotate_tensor(output, random_angle)
            noise = rotate_tensor(noise, random_angle)        
        input[:,0] = torch.clamp(input[:,0] + noise[:,0], -1, 1)        
        img_list.append(torch.cat((input,output), dim = 1))        
    data = torch.cat(img_list, dim=0)
    ins = data[:,0:3,:,:]
    outs = data[:,3:,:,:]
    return ins, outs
