from __future__ import print_function
import torch
from models import SketchModule, ShapeMatchingGAN
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch, load_style_image_pair, cropping_training_batches
import random
from vgg import get_GRAM, VGGFeature
import torchvision.models as models
from options import TrainShapeMatchingOptions
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TrainShapeMatchingOptions()
    opts = parser.parse()

    # create model
    print('--- create model ---')
    netShapeM = ShapeMatchingGAN(opts.GS_nlayers, opts.DS_nlayers, opts.GS_nf, opts.DS_nf,
                     opts.GT_nlayers, opts.DT_nlayers, opts.GT_nf, opts.DT_nf, opts.gpu)

    if opts.gpu:
        netShapeM.cuda()
    netShapeM.init_networks(weights_init)
    netShapeM.train()

    if opts.style_loss:
        netShapeM.G_S.load_state_dict(torch.load(opts.load_GS_name))  
        netShapeM.G_S.eval()
        VGGNet = models.vgg19(pretrained=True).features
        VGGfeatures = VGGFeature(VGGNet, opts.gpu)
        for param in VGGfeatures.parameters():
            param.requires_grad = False
        if opts.gpu:
            VGGfeatures.cuda()
        style_targets = get_GRAM(opts.style_name, VGGfeatures, opts.batchsize, opts.gpu)

    print('--- training ---')
    # load image pair
    _, X, Y, Noise = load_style_image_pair(opts.style_name, gpu=opts.gpu)
    Y = to_var(Y) if opts.gpu else Y
    X = to_var(X) if opts.gpu else X
    Noise = to_var(Noise) if opts.gpu else Noise
    for epoch in range(opts.texture_step1_epochs):
        for i in range(opts.Ttraining_num//opts.batchsize):
            x, y = cropping_training_batches(X, Y, Noise, opts.batchsize, 
                                      opts.Tanglejitter, opts.subimg_size, opts.subimg_size)
            losses = netShapeM.texture_one_pass(x, y)
            print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.texture_step1_epochs, i+1,
                                                         opts.Ttraining_num//opts.batchsize), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lsty: %+.3f'%(losses[0], losses[1], losses[2], losses[3])) 
    if opts.style_loss:
        fnames = load_train_batchfnames(opts.text_path, opts.batchsize, 
                                        opts.text_datasize, trainnum=opts.Ttraining_num)
        for epoch in range(opts.texture_step2_epochs):
            itr = 0
            for fname in fnames:
                itr += 1
                t = prepare_text_batch(fname, anglejitter=False)
                x, y = cropping_training_batches(X, Y, Noise, opts.batchsize, 
                                      opts.Tanglejitter, opts.subimg_size, opts.subimg_size)
                t = to_var(t) if opts.gpu else t
                losses = netShapeM.texture_one_pass(x, y, t, 0, VGGfeatures, style_targets)  
                print('Step2, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.texture_step2_epochs, 
                                                             itr, len(fnames)), end=': ')
                print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lsty: %+.3f'%(losses[0], losses[1], losses[2], losses[3])) 

    print('--- save ---')
    # directory
    netShapeM.save_texture_model(opts.save_path, opts.save_name)   

if __name__ == '__main__':
    main()
