from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
import argparse
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="camera/color/")
    parser.add_argument('--generated', default="camera/generated/")
    return parser


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    
    # tree is 0, sky is 1, ground is 2
    # We use an arbitrary .4 threshold because our classifier's predictions for ground and tree are very similar at times, tied at around .42-.44
    for i in range(pred_mask.shape[1]):
        for j in range(pred_mask.shape[2]):
            if pred_mask[0, i, j, 0] > 0.4:
                pred_mask[0, i, j, 0] = 1.0

    pred_mask = tf.argmax(pred_mask, axis=-1)

    # pred_mask = pred_mask.numpy()
    # for i in range(pred_mask.shape[1]):
    #     for j in range(pred_mask.shape[2]):
    #         if pred_mask[0, i, j] == 0:
    #             pred_mask[0, i, j] = 0
    #         elif pred_mask[0, i, j] == 1:
    #             pred_mask[0, i, j] = 29
    #         else:
    #             pred_mask[0, i, j] = 255

    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


if __name__ == "__main__":
    argparser = load_args()
    opt = argparser.parse_args()

    Path(opt.generated).mkdir(parents=True, exist_ok=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    trained_model = tf.keras.models.load_model('trav_classifier.h5')
    trained_model.summary()

    for filename in os.listdir(opt.dataset):
        filepath = opt.dataset + filename
        
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        # add 1 for batch dimensions
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32) / 255.0

        print("image: ", image.shape)
        pred_mask = trained_model.predict(image)
        print(pred_mask.shape)
        pred_mask = create_mask(pred_mask)
        # remove 1 from batch dimensions
        pred_mask = tf.squeeze(pred_mask, [0])
        
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask))
        plt.axis('off')
        # plt.show()

        filename = filename.replace("color", "gen")
        save_path = opt.generated + filename
        plt.savefig(save_path)
