from scipy.ndimage import distance_transform_edt
from skimage.measure import label
import tensorflow as tf
import numpy as np

def unet_sample_weights(x, w0=10., sigma=5., data_type=np.float32, use_distance_factor=True):
    """
    Sample weight calculation according to the U-Net paper (O Ronneberger, P Fischer, T Brox; 2015)
    
    For 2D map labels x ∈ Ω, this function returns w(x) where

    w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2/(2 * sigma^2))

    For more details: https://arxiv.org/pdf/1505.04597

    If use_distance_factor == False then w0=0
    """
    if x.dtype != np.int32: 
        x = x.astype(np.int32)
    sample_weights = np.zeros(shape=x.shape, dtype=data_type)
    img_shape = x.shape
    img_size = np.prod(img_shape)
    # Calculates the class weight w_c(x). Most frequent class has weight=1, other classes have weight > 1
    unique, idx = np.unique(x.ravel(), return_inverse=True)
    freq_classes = np.bincount(idx) / img_size # frequency of each class
    w_c = freq_classes[idx].max() / freq_classes[idx].reshape(img_shape) # normalize
    sample_weights += w_c
    # Calculates the distance factor given the two smallest distances between two foreground objects
    if use_distance_factor:
        # Labels the foreground objects
        objects_labels, num_labels = label(x, 0, True, 2)
        foreground_labels = np.unique(objects_labels)[1:]
        if num_labels > 2: # a at least 2 foreground objects are required
            maps = np.zeros(shape=(num_labels - 1, img_shape[0], img_shape[1]), dtype=data_type)
            for j, map_j in enumerate(maps):
                map_j += distance_transform_edt(objects_labels != foreground_labels[j]) # euclidean distance (edt)
            d1, d2 = np.sort(maps, axis=0)[:2] # Pick the 2 maps containing the two smallest edts
            sample_weights += w0 * np.exp(-(d1 + d2)**2. / (2. * sigma**2.))
    return sample_weights

@tf.function
def dice_loss(y_true, y_pred):
    ohe_y = tf.expand_dims(y_true, axis=-1) # one-hot-encoded y_true
    ohe_y = tf.cast(tf.concat([1 - ohe_y, ohe_y], axis=-1), y_pred.dtype)
    dice =  1.0 - 2.0 * tf.reduce_sum(y_pred * ohe_y, axis=(1,2,3)) \
                        / (tf.reduce_sum(ohe_y, axis=(1,2,3)) + tf.math.reduce_sum(y_pred, axis=(1,2,3)) + tf.keras.backend.epsilon())
    return dice
    
@tf.function
def IoU(y_true, y_pred):
    y_pred = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.bool)
    y_true = tf.cast(y_true, tf.bool)
    tp = tf.math.reduce_sum(tf.cast(tf.math.logical_and(y_true, y_pred), tf.float32), axis=(1,2)) 
    fp = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_true), y_pred), tf.float32), axis=(1,2))
    fn = tf.math.reduce_sum(tf.cast(tf.math.logical_and(tf.math.logical_not(y_pred), y_true), tf.float32), axis=(1,2))
    return tp / (tp + fp + fn + tf.keras.backend.epsilon())