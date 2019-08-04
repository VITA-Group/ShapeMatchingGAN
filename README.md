# ShapeMatchingGAN

<img src="https://github.com/williamyang1991/TET-GAN/blob/master/imgs/teaser.png" width="80%" height="80%">

This is a pytorch implementation of the paper.

Shuai Yang, Zhangyang Wang, Zhaowen Wang, Ning Xu, Jiaying Liu and Zongming Guo.

Controllable Artistic Text Style Transfer via Shape-Matching GAN, 

Accepted by International Conference on Computer Vision (ICCV), 2019.

[[Project]](https://williamyang1991.github.io/projects/ICCV2019/SMGAN.html) | [[Paper]](https://arxiv.org/abs/1905.01354) 

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
git clone https://github.com/williamyang1991/ShapeMatchingGAN.git
cd ShapeMatchingGAN/src
```
## Testing Example

- Download pre-trained models from [[Google Drive]](https://drive.google.com/open?id=1gjHR39deUSPChtRbKAD80waoQFTiXyMs) or [[Baidu Cloud]](https://pan.baidu.com/s/11LVKWAd6BCgWQqM6SZByEQ) to `../save/`
- Artisic text style transfer using <i>fire</i> style with scale 0.0
  - Results can be found in `../output/`

<img src="https://github.com/williamyang1991/TET-GAN/blob/master/imgs/example.jpg" width="50%" height="50%">

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

### Training Skecth Module G_B

- Download text dataset from [[Google Drive]](https://drive.google.com/open?id=1gjHR39deUSPChtRbKAD80waoQFTiXyMs) or [[Baidu Cloud]](https://pan.baidu.com/s/11LVKWAd6BCgWQqM6SZByEQ) to `../data/`

- Train G_B with default parameters
```
python train.py 
```
Saved model can be found at `../save/`
- Use `--help` to view more training options
```
python train.py --help
```
  
### Oneshot Training

- Download a pre-trained model from [[Google Drive]](https://drive.google.com/file/d/1pNOE4COeoXp_-N4IogNS-GavCBbZJtw1/view?usp=sharing) or [[Baidu Cloud]](https://pan.baidu.com/s/1yK6wM0famUwu25s1v92Emw) to `../save/`
  - Specify the pretrained model to load using the option `--load_model_name`

- Finetune TET-GAN on a new style/glyph image pair (supervised oneshot training)
```
python oneshotfinetune.py --style_name ../data/oneshotstyle/1-train.png
```
Saved model can be found at `../save/`
- Finetune TET-GAN on a new style image without its glyph counterpart (unsupervised oneshot training)
```
python oneshotfinetune.py --style_name ../data/oneshotstyle/1-train.png --supervise 0
```
Saved model can be found at `../save/`
- Use `--help` to view more finetuning options
```
python oneshotfinetune.py --help
```

### Contact

Shuai Yang

williamyang@pku.edu.cn
