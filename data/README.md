## Dataset information
### Data Download
- Text dataset for training Sketch Module can be downloaded from [[Google Drive]](https://drive.google.com/open?id=1gjHR39deUSPChtRbKAD80waoQFTiXyMs) or [[Baidu Cloud]](https://pan.baidu.com/s/11LVKWAd6BCgWQqM6SZByEQ)
### Images are arranged in this way 
#### 708 Chinese characters for training
```
rawtext/yaheiB/train/0000.png
rawtext/yaheiB/train/0001.png
...
rawtext/yaheiB/train/0706.png
rawtext/yaheiB/train/0707.png
```
#### 76 Chinese characters, 10 Arabic numerals and 52 English letters for testing
```
rawtext/yaheiB/val/0708.png
rawtext/yaheiB/val/0709.png
...
rawtext/yaheiB/val/0835.png
rawtext/yaheiB/val/0836.png
```
#### 5 augmented images for robust training
```
rawtext/augment/0000.png
...
rawtext/augment/0004.png
```
#### style images ((X,Y) pair) for training Shape Matching GAN 
```
style/fire.png
...
style/maple.png
```
