# ShapeMatchingGAN

<table border="0" width='100%'>
 <tr align="center">
  <td width="14.5%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-a.jpg" width="100%" ></td>
  <td width="32%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-b.png" width="99%" ></td>
  <td width="33%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-c.gif" width="99%" ></td>	
  <td width="18.6%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-d.gif" width="99%" ></td>
 </tr>
 <tr align="center">
  <td>source</td><td>adjustable stylistic degree of glyph</td><td>stylized text</td><td>application</td>
</tr>					 
 </table>
 <table border="0" width='100%'>
 <tr align="center">
  <td width="50%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-e.gif" alt="" width="99%" ></td>	
  <td width="50%"><img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/teaser-f.gif" alt="" width="99%" ></td>						
 </tr>					 
 <tr align="center">
  <td>liquid artistic text rendering</td><td>smoke artistic text rendering</td>
</tr>	
</table>

This is a pytorch implementation of the paper.

Shuai Yang, Zhangyang Wang, Zhaowen Wang, Ning Xu, Jiaying Liu and Zongming Guo.
Controllable Artistic Text Style Transfer via Shape-Matching GAN, 
accepted by International Conference on Computer Vision (ICCV), 2019.

[[Project]](https://williamyang1991.github.io/projects/ICCV2019/SMGAN.html) | [[Paper]](https://arxiv.org/abs/1905.01354) | More about artistic text style transfer [[Link]](https://williamyang1991.github.io/index.html#publications)

It is provided for educational/research purpose only. Please consider citing our paper if you find the software useful for your work.


## Usage: 

#### Prerequisites
- Python 2.7
- Pytorch 1.1.0
- matplotlib
- scipy
- Pillow

#### Install
- Clone this repo:
```
git clone https://github.com/TAMU-VITA/ShapeMatchingGAN.git
cd ShapeMatchingGAN/src
```
## Testing Example

- Download pre-trained models from [[Google Drive]](https://drive.google.com/open?id=1gjHR39deUSPChtRbKAD80waoQFTiXyMs) or [[Baidu Cloud]](https://pan.baidu.com/s/1tvTCig4OQhixX_EzSzYQSg)(code:rjpi) to `../save/`
- Artisic text style transfer using <i>fire</i> style with scale 0.0
  - Results can be found in `../output/`

<img src="https://github.com/williamyang1991/ShapeMatchingGAN/blob/master/imgs/test.jpg" width="60%" height="60%">

```
python test.py \
--scale 0.0
--structure_model ../save/fire-GS-iccv.ckpt \
--texture_model ../save/fire-GT-iccv.ckpt \
--gpu
```
- Artisic text style transfer with specified parameters
  - setting scale to -1 means testing with multiple scales in \[0,1\] with step of scale_step
  - specify the input text name, output image path and name with text_name, result_dir and name, respectively
```
python test.py \
--text_name ../data/rawtext/yaheiB/val/0801.png \
--scale -1 --scale_step 0.2 \
--structure_model ../save/fire-GS-iccv.ckpt \
--texture_model ../save/fire-GT-iccv.ckpt \
--result_dir ../output --name fire-0801 \
--gpu
```
or just modifying and running
```
sh ../script/launch_test.sh
```
- For black and white text images, use option `--text_type 1`
  - utils.text_image_preprocessing will transform BW images into distance-based images
  - distance-based images make the network better deal with the saturated regions

## Training Examples

### Training Sketch Module G_B

- Download text dataset from [[Google Drive]](https://drive.google.com/open?id=1gjHR39deUSPChtRbKAD80waoQFTiXyMs) or [[Baidu Cloud]](https://pan.baidu.com/s/1tvTCig4OQhixX_EzSzYQSg)(code:rjpi) to `../data/`

- Train G_B with default parameters
  - Adding augmented images to the training set can make G_B more robust
```
python trainSketchModule.py \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--augment_text_path ../data/rawtext/augment --augment_text_datasize 5 \
--batchsize 16 --Btraining_num 12800 \
--save_GB_name ../save/GB.ckpt \
--gpu
```
or just modifying and running
```
sh ../script/launch_SketchModule.sh
```
Saved model can be found at `../save/`
- Use `--help` to view more training options
```
python trainSketchModule.py --help
```
  
### Training Structure Transfer G_S

- Train G_S with default parameters
  - step1: G_S is first trained with a fixed <i>l</i> = 1 to learn the greatest deformation
  - step2: we then use <i>l</i> ∈ {0, 1} to learn two extremes
  - step3: G_S is tuned on <i>l</i> ∈ {i/K}, i=0,...,K where K = 3 (i.e. --scale_num 4)
  - for structure with directional patterns, training without `--Sanglejitter` will be a good option
```
python trainStructureTransfer.py \
--style_name ../data/style/fire.png \
--batchsize 16 --Straining_num 2560 \
--step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
--scale_num 4 \
--Sanglejitter \
--save_path ../save --save_name fire \
--gpu
```
or just modifying and running
```
sh ../script/launch_ShapeMGAN_structure.sh
```
Saved model can be found at `../save/`
- To preserve the glyph legibility (Eq. (7) in the paper), use option `--glyph_preserve`
  - need to specify the text dataset `--text_path ../data/rawtext/yaheiB/train` and `--text_datasize 708`
  - need to load pre-trained G_B model `--load_GB_name ../save/GB-iccv.ckpt`
  - in most cases, `--glyph_preserve` is not necessary, since one can alternatively use a smaller <i>l</i>
- Use `--help` to view more training options
```
python trainStructureTransfer.py --help
```

### Training Texture Transfer G_T

- Train G_T with default parameters
  - for complicated style or style with directional patterns, training without `--Tanglejitter` will be a good option
```
python trainTextureTransfer.py \
--style_name ../data/style/fire.png \
--batchsize 4 --Ttraining_num 800 \
--texture_step1_epochs 40 \
--Tanglejitter \
--save_path ../save --save_name fire \
--gpu
```
or just modifying and running
```
sh ../script/launch_ShapeMGAN_texture.sh
```
Saved model can be found at `../save/`
- To train with style loss, use option `--style_loss`
  - need to specify the text dataset `--text_path ../data/rawtext/yaheiB/train` and `--text_datasize 708`
  - need to load pre-trained G_S model `--load_GS_name ../save/fire-GS.ckpt`
  - adding `--style_loss` can slightly improve the texture details
- Use `--help` to view more training options
```
python trainTextureTransfer.py --help
```

### More

Three training examples are in the IPythonNotebook ShapeMatchingGAN.ipynb

Have fun :-)

### Try with your own style images

- Style image preparation 
  - Applicable style types: To make the stylized text easy to recognize, it is desirable to have a certain distinction between the text and the background. If the texture has no distinct shape, the generated stylized text will be mixed with the background. Therefore, textures with distinct shapes as the reference style are recommended.
  - Prepare (X,Y): Use Image Matting Algorithm or the Quick Selection Tool in Photoshop to obtain the black and white structure map X (i.e. foreground mask) of the style image Y. 
  - Prepare distance-based structure map: Use utils.text_image_preprocessing to transform black and white X into distance-based X.
  - Concatenate distance-based X with Y as the format of images in `../data/style/` and copy the result to `../data/style/`.

<img src="https://github.com/TAMU-VITA/ShapeMatchingGAN/blob/master/imgs/failure.jpg" width="90%" height="90%">

### Contact

Shuai Yang

williamyang@pku.edu.cn
