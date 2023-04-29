# T-former: An Efficient Transformer for Image Inpainting
This is the code for ACM multimedia 2022 “T-former: An Efficient Transformer for Image Inpainting”
# train:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-4
# fine_tune:
python train.py --no_flip --no_rotation --no_augment --image_file your_data --lr 1e-5 --continue_train
# test:
python test.py --mask_type 3 --image_file your_data --mask_file your_mask --continue_train
