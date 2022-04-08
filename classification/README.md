## Training
The provided code is an example of applying YOCO to horizontal flip. Other augmentations are similar. 
Mixup is written as comments inside the main.py. We use 4 gpus and 2 gpus for training on ImageNet and CIFAR, respectively.

### ImageNet
Run with YOCO
```
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 -y \
[your imagenet-folder with train and val folders]
```
or image-level aug
```
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
[your imagenet-folder with train and val folders]
```
### CIFAR-100
First
```
cd cifar
```
Run with YOCO
```
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 -y
```
or image-level aug
```
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 
```

## Evaluation

For partial image recognition, run 
```
python -W ignore main.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --evaluate2 --world-size 1 --rank 0 \
--resume [your path to pretrained model] \
[your imagenet-folder with train and val folders]
```

For calibration, see [PixMix](https://github.com/andyzoujm/pixmix).

For adversarial attacks, see [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch). 

For corruption robustness, see [Co-Mixup](https://github.com/snu-mllab/Co-Mixup).

For ImageNet-A, see [Natural Adversarial Examples](https://github.com/hendrycks/natural-adv-examples).

## Pre-trained models

We provide our pre-trained YOCO ImageNet classification models at: https://drive.google.com/drive/folders/1CmZW6UoR-YESqQ00IMPE9fVDfYjBa5IZ?usp=sharing


