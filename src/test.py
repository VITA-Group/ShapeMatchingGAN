from options import TestOptions
import torch
from models import GlyphGenerator, TextureGenerator
from utils import load_image, to_data, to_var, visualize, save_image, gaussian
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    # data loader
    print('--- load data ---')
    text = load_image(opts.text_name, opts.text_type)
    label = opts.scale
    step = opts.scale_step * 2.0
    if opts.gpu:
        text = to_var(text)
    
    # model
    print('--- load model ---')
    netGlyph = GlyphGenerator(n_layers=6, ngf=32)
    netTexture = TextureGenerator(n_layers=6)
    netGlyph.load_state_dict(torch.load(opts.structure_model))
    netTexture.load_state_dict(torch.load(opts.texture_model))
    if opts.gpu:
        netGlyph.cuda()
        netTexture.cuda()
    netGlyph.eval()
    netTexture.eval()
    
    print('--- testing ---')
    text[:,0:1] = gaussian(text[:,0:1], stddev=0.2)
    if label == -1:
        scale = -1.0
        noise = text.data.new(text[:,0:1].size()).normal_(0, 0.2)
        result = []
        while scale <= 1.0: 
            img_str = netGlyph(text, scale) 
            img_str[:,0:1] = torch.clamp(img_str[:,0:1] + noise, -1, 1)
            result1 = netTexture(img_str).detach()
            result = result + [result1]
            scale = scale + step
    else:
        img_str = netGlyph(text, label*2.0-1.0) 
        img_str[:,0:1] = gaussian(img_str[:,0:1], stddev=0.2)
        result = [netTexture(img_str)]
    
    if opts.gpu:
        for i in range(len(result)):              
            result[i] = to_data(result[i])
        
    print('--- save ---')
    # directory
    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)         
    for i in range(len(result)):     
        if label == -1:
            result_filename = os.path.join(opts.result_dir, (opts.name+'_'+str(i)+'.png'))
        else:
            result_filename = os.path.join(opts.result_dir, (opts.name+'.png'))
        save_image(result[i][0], result_filename)

if __name__ == '__main__':
    main()
