# Intel & MobileODT Cervical Cancer Screening, kaggle competition
## Dependencies
* [pytorch 0.1.12](http://pytorch.org/), installed via pip or from source
* [torchvision](https://github.com/pytorch/vision.git), installed from source
* [torchsample, commit 00a41b05](https://github.com/ncullen93/torchsample/commit/00a41b05d43b2815d915c632c94471e208e90153)
* `pip install tensorboard_logger, tqdm`
* tensorflow >= 1.0 to use tensorboard (cpu version is enough)
* other necessary dependencies...

## Steps to reproduce our submissions

### Prepare data:
The folder `./data_utils` contains codes to generate index files (all the idx files used for our models have been generated in `./data_utils`) for train/val/test. Note that the images should be put following our folder structure:
```
# data_root
# ├── additional
# │   ├── Type_1
# │   ├── Type_2
# │   └── Type_3
# ├── test
# │   └── test
# └── train
#     ├── Type_1
#     ├── Type_2
#     └── Type_3
```

We resize all the original images to 256x256, please use `data_utils/resize_convert_all_multi.py` and change the paths in it to generate the resized data.
We assume the resized_data root is `./data/resized_data/`

### train:
Use `./scripts/train_10_fold.py` and change the `arch` to `resnet101` and `inception_v3` to train two kinds of models, respectively.

In this step, we will get model checkpoints in 20 different sub-folders in `./ckpt`.


### make submission (10-crop, 10-fold ensemble, version 1):
- **10 crop test** for each single model: use `./scripts/gen_test_10_fold_cmds.py` and change the `arch` to `resnet101` and `inception_v3`, respectively. We will get 20 different submission files in `./submission`, each of which corresponds to a single model.

- ensemble scores of all models (average): put all the paths of the submission files in a file, for example:
 ```
 ls submission/resnet101_model_epoch_15_fold_* > resnet101_epoch15_10_fold.txt
 ls submission/inception_v3_model_epoch_15_fold_* > inception_v3_epoch15_10_fold.txt
 cat resnet101_epoch15_10_fold.txt inception_v3_epoch15_10_fold.txt > inception_v3_resnet101_epoch15_10_folds.txt
 python ensemble_scores.py inception_v3_resnet101_epoch15_10_folds.txt
 ```

The final submission file is like `./submission/ensemble_submission_inception_v3_resnet101_epoch15_10_folds__{date_time}.txt`
(This scores 0.48662 on public Leaderboard, and 0.85024 on the private Leaderboard.)


### make submission (10-crop + 40-crop, 10-fold ensemble, version 2):
- **10 crop test** for each single model: use `./scripts/gen_test_10_fold_cmds.py` and change the `arch` to `resnet101` and `inception_v3`, respectively. We will get 20 different submission files in `./submission`, each of which corresponds to a single model.
(Same as version 1.)

- **40 crop test** for each single model: use **`./scripts/gen_test_10_fold_cmds_40_crop.py`** and change the `arch` to `resnet101` and `inception_v3`, respectively. We will get 20 different submission files in `./submission`, each of which corresponds to a single model.


- ensemble scores of all models (average): put all the paths of the submission files in a file, for example:

```Shell
 # 10 crop
 ls submission/resnet101_model_epoch_15_fold_* > resnet101_epoch15_10_fold.txt
 ls submission/inception_v3_model_epoch_15_fold_* > inception_v3_epoch15_10_fold.txt
 cat resnet101_epoch15_10_fold.txt inception_v3_epoch15_10_fold.txt > inception_v3_resnet101_epoch15_10_folds.txt
 python ensemble_scores.py inception_v3_resnet101_epoch15_10_folds.txt

 # 40 crop
 ls submission/resnet101_model_epoch_15_fold_*_40* > 40crop_resnet101_epoch15_10_fold.txt
 ls submission/inception_v3_model_epoch_15_fold_*_40* > 40crop_inception_v3_epoch15_10_fold.txt
 cat 40crop_resnet101_epoch15_10_fold.txt 40crop_inception_v3_epoch15_10_fold.txt > 40crop_inception_v3_resnet101_epoch15_10_folds.txt
 python ensemble_scores.py 40crop_inception_v3_resnet101_epoch15_10_folds.txt

 # merge 10 and 40 crop (**NB**: 10crop is used twice)
 ls submission/ensemble_submission_inception_v3_resnet101_epoch15_10_folds*  submission/ensemble_submission_inception_v3_resnet101_epoch15_10_folds*  submission/ensemble_submission_40crop_inception_v3_resnet101_epoch15_10_folds* > 10_40_crop_res101_v3_epoch15.txt
 python ensemble_scores.py 10_40_crop_res101_v3_epoch15.txt
```

The final submission file is like `./submission/ensemble_submission_10_40_crop_res101_v3_epoch15__{date_time}.txt`
(This scores 0.48592 on the public Leaderboard, and 0.83833 on the private Leaderboard.)

## Credits
Written and maintained by:
* [Gu Wang](https://github.com/wangg12)
* [Yu Yang](https://github.com/yangyu12)
* [Shi Yan](https://github.com/neycyanshi)
* [Junjie Wu](https://github.com/THUwu)
