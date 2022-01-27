
## MoCo: Transferring to Detection

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

In YOCO, please note we change the number of GPUs from 8 to 4.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained MoCo model to detectron2's format:
   ```
   python3 convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_moco.yaml \
	--num-gpus 4 MODEL.WEIGHTS ./output.pkl
   ```

