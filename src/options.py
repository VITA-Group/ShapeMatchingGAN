import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--text_name', type=str, default='../data/rawtext/yaheiB/val/0801.png', help='path of the text image')
        self.parser.add_argument('--scale', type=float, default=0.0, help='glyph deformation degree,  0~1 for single scale, -1 for multiple scales in 0~1 with step of scale_step')
        self.parser.add_argument('--scale_step', type=float, default=0.2, help='scale step')
        self.parser.add_argument('--text_type', type=int, default=0, help='0 for distance-based text image, 1 for black and white text image')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='output', help='file name of the outputs')
        self.parser.add_argument('--result_dir', type=str, default='../output/', help='path for saving result images')

        # model related
        self.parser.add_argument('--structure_model', type=str, default='../save/fire-GS-iccv.ckpt', help='specified the dir of saved structure transfer models')
        self.parser.add_argument('--texture_model', type=str, default='../save/fire-GT-iccv.ckpt', help='specified the dir of saved texture transfer models')
        self.parser.add_argument('--gpu', action='store_true', default=False, help='Whether using gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

class TrainSketchOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## Sketch Module
        # data loader related
        self.parser.add_argument('--text_path', type=str, default='../data/rawtext/yaheiB/train', help='path of the text images')
        self.parser.add_argument('--augment_text_path', type=str, default='../data/rawtext/augment/', help='path of the augmented text images')        
        self.parser.add_argument('--text_datasize', type=int, default=708, help='how many text images are loaded for training')
        self.parser.add_argument('--augment_text_datasize', type=int, default=5, help='how many augmented text images are loaded for training')

        # ouptput related
        self.parser.add_argument('--save_GB_name', type=str, default='../save/GB.ckpt', help='path of the trained model to be saved')
        
        # model related
        self.parser.add_argument('--GB_nlayers', type=int, default=6, help='number of layers in Generator')
        self.parser.add_argument('--DB_nlayers', type=int, default=5, help='number of layers in Discriminator')  
        self.parser.add_argument('--GB_nf', type=int, default=32, help='number of features in the first layer of the Generator')
        self.parser.add_argument('--DB_nf', type=int, default=32, help='number of features in the first layer of the Discriminator')          
        # trainingg related
        self.parser.add_argument('--epochs', type=int, default=3, help='epoch number')
        self.parser.add_argument('--batchsize', type=int, default=16, help='batch size')
        self.parser.add_argument('--Btraining_num', type=int, default=12800, help='how many training images in each epoch')
        self.parser.add_argument('--gpu', action='store_true', default=False, help='Whether using gpu')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt    
    
    
class TrainShapeMatchingOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        ## Structure and Texture 
        # data loader realted 
        self.parser.add_argument('--style_name', type=str, default='../data/style/fire.png', help='path of the style image')   
        self.parser.add_argument('--load_GS_name', type=str, default='../save/fire-GS.ckpt', help='path to load the saved G_S model when training G_T')   
        self.parser.add_argument('--Sanglejitter', action='store_true', default=False, help='Whether rotating the training images when training G_S')          
        self.parser.add_argument('--Tanglejitter', action='store_true', default=False, help='Whether rotating the training images when training G_T')    
        # ouptput related
        self.parser.add_argument('--save_path', type=str, default='../save/', help='dir of the trained model to be saved')   
        self.parser.add_argument('--save_name', type=str, default='fire', help='fikename of the trained model to be saved')          
        # model related
        self.parser.add_argument('--GS_nlayers', type=int, default=6, help='number of layers in structure generator G_S')
        self.parser.add_argument('--DS_nlayers', type=int, default=4, help='number of layers in structure discriminator D_S')  
        self.parser.add_argument('--GS_nf', type=int, default=32, help='number of features in the first layer of G_S')
        self.parser.add_argument('--DS_nf', type=int, default=32, help='number of features in the first layer of D_S') 
        self.parser.add_argument('--GT_nlayers', type=int, default=6, help='number of layers in texture generator G_T')
        self.parser.add_argument('--DT_nlayers', type=int, default=4, help='number of layers in texture discriminator D_T')  
        self.parser.add_argument('--GT_nf', type=int, default=32, help='number of features in the first layer of G_T')
        self.parser.add_argument('--DT_nf', type=int, default=32, help='number of features in the first layer of D_T')                      
        # trainingg related
        self.parser.add_argument('--scale_num', type=int, default=4, help='how many scales are uniformly sampled between [0,1]')
        self.parser.add_argument('--step1_epochs', type=int, default=30, help='epoch number of training G_S on the max scale {1}')
        self.parser.add_argument('--step2_epochs', type=int, default=40, help='epoch number of training G_S on the two extreme scales {0, 1}') 
        self.parser.add_argument('--step3_epochs', type=int, default=80, help='epoch number of training G_S on the full scales')     
        self.parser.add_argument('--step4_epochs', type=int, default=10, help='epoch number of training G_S with glyph loss')
        self.parser.add_argument('--texture_step1_epochs', type=int, default=40, help='epoch number of training G_T without style loss')
        self.parser.add_argument('--texture_step2_epochs', type=int, default=10, help='epoch number of training G_T with style loss')
        self.parser.add_argument('--batchsize', type=int, default=16, help='batch size')
        self.parser.add_argument('--subimg_size', type=int, default=256, help='size of sub-images, which are cropped from a single image to form a training set')
        self.parser.add_argument('--Straining_num', type=int, default=2560, help='how many training images in each epoch for strcture transfer')        
        self.parser.add_argument('--Ttraining_num', type=int, default=800, help='how many training images in each epoch for texture transfer')
        self.parser.add_argument('--glyph_preserve', action='store_true', default=False, help='Whether using glyph loss to preserve the text legibility')        
        self.parser.add_argument('--style_loss', action='store_true', default=False, help='Whether using style loss')
        self.parser.add_argument('--gpu', action='store_true', default=False, help='Whether using gpu')

        ## Sketch Module
        # data loader related
        self.parser.add_argument('--text_path', type=str, default='../data/rawtext/yaheiB/train', help='path of the text images')     
        self.parser.add_argument('--text_datasize', type=int, default=708, help='how many text images are loaded for training')
        # model related
        self.parser.add_argument('--GB_nlayers', type=int, default=6, help='number of layers in Generator G_B')
        self.parser.add_argument('--DB_nlayers', type=int, default=5, help='number of layers in Discriminator D_B')  
        self.parser.add_argument('--GB_nf', type=int, default=32, help='number of features in the first layer of G_B')
        self.parser.add_argument('--DB_nf', type=int, default=32, help='number of features in the first layer of D_B') 
        self.parser.add_argument('--load_GB_name', type=str, default='../save/GB-iccv.ckpt', help='specified the dir of saved Sketch Module')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt    