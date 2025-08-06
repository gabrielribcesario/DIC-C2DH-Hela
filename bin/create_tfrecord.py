from natsort import natsorted
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse
import os

# e.g. img_dir="DIC-C2DH-HeLa-Train/01", img_shape=(512, 512, 1), img_mode=cv.IMREAD_UNCHANGED
def get_images(img_dir, img_shape, img_mode, dtype='uint8'):
    img_list = natsorted(os.listdir(img_dir))
    X = np.zeros((len(img_list),) + img_shape, dtype=dtype)
    for i, img_path in enumerate(img_list):
        img = cv.imread(f"{img_dir}/{img_path}", img_mode)
        if img.shape != img_shape: img = img.reshape(img_shape)
        X[i] += img.astype(dtype)
    return X

# e.g. img_dir="DIC-C2DH-HeLa-Train/01", img_shape=(512, 512, 1), img_mode=cv.IMREAD_UNCHANGED
def get_masks(mask_dir, mask_shape, mask_mode, dtype='int32'): 
    mask_list = natsorted(os.listdir(mask_dir))
    y = np.zeros((len(mask_list),) + mask_shape, dtype=dtype)
    for i, mask_path in enumerate(mask_list):
        mask = cv.imread(f"{mask_dir}/{mask_path}", mask_mode) != 0
        if mask.shape != mask_shape: mask = mask.reshape(img_shape) # if theres is an extra/missing axis
        y[i] += mask.astype(dtype)
    return y

def get_dataset(dirname: str, seg: int, mask: bool, 
                img_shape: tuple[int, int, int], img_mode: int,
                mask_shape: tuple[int, int], mask_mode: int):
    # Get image-mask pairs from segment <seg> at <dirname>
    ds = {}
    ds["image"] = get_images(f"{dirname}/0{seg}", img_shape, img_mode, 'uint8')
    if mask: # Use the less noisy, more complete target masks from ERR_SEG
        ds["mask"]= get_masks(f"{dirname}/0{seg}_ERR_SEG", mask_shape, mask_mode, 'int32')
    return ds

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str)
parser.add_argument('--test', type=str)
parser.add_argument('--tf', type=str)
args = parser.parse_args()
train_dir = args.train # Read training data from here
test_dir = args.test # Read test data from here
tf_dir = args.tf # Store both Tensorflow datasets here

if tf_dir is None:
    tf_dir = "TFData"

os.chdir("..")
os.system(f"mkdir -p {tf_dir}")

img_shape, img_mode = (512, 512, 1), cv.IMREAD_UNCHANGED
mask_shape, mask_mode = (512, 512), cv.IMREAD_UNCHANGED

# Split the segments 01 and 02 for cross-validation
if train_dir is not None:
    print("Processing training data...")
    train_ds = {"01": tf.data.Dataset.from_tensor_slices(get_dataset(train_dir, 1, True, img_shape, img_mode, mask_shape, mask_mode)),
                "02": tf.data.Dataset.from_tensor_slices(get_dataset(train_dir, 2, True, img_shape, img_mode, mask_shape, mask_mode)) } 
    print("Adding training dataset to .tfrecord...")
    tr_builder=  tfds.dataset_builders.store_as_tfds_dataset(name='hela_train',
                                                             version="1.0.0",
                                                             features=tfds.features.FeaturesDict({'image': tfds.features.Tensor(shape=img_shape, dtype=np.uint8), 
                                                                                                   'mask': tfds.features.Tensor(shape=mask_shape, dtype=np.int32)}),
                                                             data_dir=tf_dir,
                                                             split_datasets=train_ds,
                                                             disable_shuffling=True) 
    tr_builder.download_and_prepare()
    print(".tfrecord created.")

# Split the segments 01 and 02 for prediction
if test_dir is not None:
    print("Processing test data...")
    test_ds = {"01": tf.data.Dataset.from_tensor_slices(get_dataset(test_dir, 1, False, img_shape, img_mode, mask_shape, mask_mode)),
               "02": tf.data.Dataset.from_tensor_slices(get_dataset(test_dir, 2, False, img_shape, img_mode, mask_shape, mask_mode)) }
    print("Adding test dataset to .tfrecord...")
    te_builder=  tfds.dataset_builders.store_as_tfds_dataset(name='hela_test', # different file name since mask is missing 
                                                             version="1.0.0",
                                                             features=tfds.features.FeaturesDict({'image': tfds.features.Tensor(shape=img_shape, dtype=np.uint8)}),
                                                             data_dir=tf_dir,
                                                             split_datasets=test_ds,
                                                             disable_shuffling=True)
    te_builder.download_and_prepare()
    print(".tfrecord created.")