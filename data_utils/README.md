This folder contains some data utils code, 
since the original dataset may have some problems according to kagglers' discussions and kernels.

### run
* `python check_data.py`: check data
* `python gen_all_list.py`: generate `train_total.csv, additional.csv, train_additional.csv, test.csv`
* `python gen_train_val_idx.py`: generate train/val idx
* `python resize_convert_all_multi.py`: resize images, store the images in new directory, and keep the folder structure.

### description of files
* `train_total.csv`: `<relative_image_path>,<Type_str>,<int_label>`. Images from the `train` folder. `Type_str: int_label`: `Type_1:0, Type_2:1, Type_3:2`.
* `additional.csv`: Images from the `additional` folder.
* `train_additional.csv`: Images from the `train` and `additional` folders.
* `bad_images.csv`: `<relative_image_path>,<bad_type>`. `bad_type: 0_byte, truncated`.

* `train_idx.csv`: Train images after split cleaned training data.
* `val_idx.csv`: Validation images after split cleaned validating data.
* `test.csv`: Test images.