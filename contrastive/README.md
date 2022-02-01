## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning


This repo is MoCo v2 + YOCO. We only modify main_moco.py and ./moco/loader.py. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>

Cite the [MoCo paper](https://arxiv.org/abs/1911.05722):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
or the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```


### Preparation

Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  [your imagenet-folder with train and val folders]
```
This script uses all the default hyper-parameters as described in the MoCo v2 paper.

***Note***: for 4-gpu training, we recommend following the [linear lr scaling recipe](https://arxiv.org/abs/1706.02677): `--lr 0.015 --batch-size 128` with 4 gpus. We got similar results using this setting.


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  [your imagenet-folder with train and val folders]
```


### Transferring to Object Detection

See [./detection](detection).

## Pre-trained models

We provide our pre-trained YOCO contrastive learning pre-trained models at: https://drive.google.com/drive/folders/1CmZW6UoR-YESqQ00IMPE9fVDfYjBa5IZ?usp=sharing

### License

MoCo is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also
* [moco.tensorflow](https://github.com/ppwwyyxx/moco.tensorflow): A TensorFlow re-implementation.
* [Colab notebook](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb): CIFAR demo on Colab GPU.
