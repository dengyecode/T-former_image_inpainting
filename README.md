# T-former: An Efficient Transformer for Image Inpainting (MM 2022)
This is the code for ACM multimedia 2022 “T-former: An Efficient Transformer for Image Inpainting”
# train:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-4
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-5 --continue_train
# test:
python test.py --mask_type 3 --image_file your_data --mask_file your_mask --continue_train


## Citation
If you are interested in this work, please consider citing:

    @inproceedings{tformer_image_inpainting,
      author = {Deng, Ye and Hui, Siqi and Zhou, Sanping and Meng, Deyu and Wang, Jinjun},
      title = {T-former: An Efficient Transformer for Image Inpainting},
      year = {2022},
      isbn = {9781450392037},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      doi = {10.1145/3503161.3548446},
      pages = {6559–6568},
      numpages = {10},
      series = {MM '22}
}

