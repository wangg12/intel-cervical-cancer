# Faster RCNN Detector with PyTorch
This is modified [PyTorch](https://github.com/pytorch/pytorch)
implementation of Faster RCNN. 
This project is mainly based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
and [TFFRCNN](https://github.com/CharlesShang/TFFRCNN).

For details about R-CNN please refer to the [paper](https://arxiv.org/abs/1506.01497) 
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks 
by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### TODO

- [x] Add IntelMobileODT_Cervical_Cancer_Screening data/bbx API
- [x] Load model from torchvision.models.net, save as h5df
- [x] Add optional feature extractors, extend compatibility with futer params sharing strategy.
- [x] Evaluation
- [ ] Visualization training and validation on tensorboard
- [ ] Train on all annotations, maybe need more bbxs.
- [ ] Hyper params selection and finetuning
- [ ] Connect with classification network C
- [ ] Share params with C
- [ ] Save model as pth(optional)

### Installation and demo
1. Fetch the faster_rcnn_detector branch
2. Build the Cython modules for nms and the roi_pooling layer
    ```bash
    cd faster_rcnn_pytorch/faster_rcnn
    ./make.sh
    ```
3. (Optional) Download the trained model [VGGnet_fast_rcnn_iter_70000.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WOXdpYVFybWxiZFE) 
and set the model path in `demo.py`
4. Run demo `python demo.py`

### Training on IntelMobileODT_Cervical_Cancer_Screening

1. (still messy)Please make sure you have already set up a voc-like tree structure of your dataset(in your `py-faster-rcnn/data/` folder):
```
    cervix
    ├── additional
    │   ├── Type_1
    │   ├── Type_2
    │   └── Type_3
    ├── Annotations
    ├── ImageSets
    ├── test
    │   └── test
    └── train
        ├── Type_1
        ├── Type_2
        └── Type_3
```

2. Create symlinks for the IntelMobileODT_Cervical_Cancer_Screening dataset, and put folder `Annotation` in `data/cervix/`
```bash
cd faster_rcnn_pytorch
mkdir data
ln -s $IntelMobileODT_Cervical_Cancer_Screening cervix
mv Annotation data/cervix/
```

3. Download pre-trained model: 2 options
    * download and import models from torchvision
    * [VGG16](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path `./data/pretrain_model/VGG_imagenet.npy`

4. Then you can set some hyper-parameters in `train.py` and training parameters in the `.yml` file.

5. You may need to tune the loss function defined in `faster_rcnn/faster_rcnn.py` by yourself.

### Training with TensorBoard
With the aid of [Crayon](https://github.com/torrvision/crayon),
we can access the visualisation power of TensorBoard for any 
deep learning framework.

To use the TensorBoard, install Crayon (https://github.com/torrvision/crayon)
and set `use_tensorboard = True` in `faster_rcnn/train.py`.

### Evaluation
Set the path of the trained model in `test.py`.
```bash
cd faster_rcnn_pytorch
mkdir output
python test.py
```

License: MIT license (MIT)
