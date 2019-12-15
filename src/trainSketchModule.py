from __future__ import print_function
import torch
from models import SketchModule
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
from utils import load_train_batchfnames, prepare_text_batch
from options import TrainSketchOptions
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # parse options
    parser = TrainSketchOptions()
    opts = parser.parse()

    # create model
    print('--- create model ---')
    netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)
    if opts.gpu:
        netSketch.cuda()
    netSketch.init_networks(weights_init)
    netSketch.train()

    print('--- training ---')
    for epoch in range(opts.epochs):
        itr = 0
        fnames = load_train_batchfnames(opts.text_path, opts.batchsize, 
                                        opts.text_datasize, trainnum=opts.Btraining_num)
        fnames2 = load_train_batchfnames(opts.augment_text_path, opts.batchsize, 
                                        opts.augment_text_datasize, trainnum=opts.Btraining_num)
        for ii in range(len(fnames)):
            fnames[ii][0:opts.batchsize//2-1] = fnames2[ii][0:opts.batchsize//2-1]
        for fname in fnames:
            itr += 1
            t = prepare_text_batch(fname, anglejitter=True)
            t = to_var(t) if opts.gpu else t
            losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])      
            print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs,itr,len(fnames)), end=': ')
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    print('--- save ---')
    # directory
    torch.save(netSketch.state_dict(), opts.save_GB_name)    

if __name__ == '__main__':
    main()
