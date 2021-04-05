import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# main class for perception
# performs semantic segmentation on the Unreal Scene
class Perception_Module:
    def __init__(self):
        # init GPU availability
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.gpus = tf.config.experimental.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(self.gpu, True)
                self.logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(self.gpus), "Physical GPUs,", len(self.logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)
        
        # non-traversable is 0, sky is 1, traversable is 2
        self.mappings = {0: np.array([[0, 0, 0]]), 1: np.array([[0, 0, 255]]), 2: np.array([[255, 255, 255]])}
        self.model = tf.keras.models.load_model('trav_classifier.h5')
        self.image = None
                    
    # image is a 1xHxWx3 (channels-last) numpy array
    def create_mask(self, image):
        image = image / 255.0
        self.image = self.model.predict(tf.convert_to_tensor(image, dtype=tf.float32)) 
        print(image.shape)

        for i in range(self.image.shape[1]):
            for j in range(self.image.shape[2]):
                if self.image[0, i, j, 0] > 0.4:
                    self.image[0, i, j, 0] = 1.0

        self.image = tf.argmax(self.image, axis=-1)
        self.image = tf.expand_dims(self.image, axis=-1)

        converted_array = self.convert_to_RGB()
        return converted_array

    
    # helper function to convert class prediction mask back to a numpy RGB array
    def convert_to_RGB(self):
        print(self.image.shape)
        self.image = self.image.numpy()
        converted_array = np.zeros((self.image.shape[1], self.image.shape[2], 3))

        for i in range(self.image.shape[1]):
            for j in range(self.image.shape[2]):
                converted_array[i, j, :] = self.mappings[self.image[0, i, j, 0]]

        # plt.imshow(converted_array)
        # plt.axis('off')
        # plt.show()
        
        converted_array = tf.expand_dims(converted_array, axis=0)
        return converted_array
        
# if __name__ == "__main__":
#     test = perception_module()
#     filepath = "camera/color/color_000001.png"
#     image = tf.io.read_file(filepath)
#     image = tf.image.decode_png(image, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.uint8)
#     # add 1 for batch dimensions
#     image = tf.expand_dims(image, axis=0)
#     image = image.numpy()

#     # pred_mask = test.model.predict(image)
#     pred_mask = test.create_mask(image)
#     print(pred_mask.shape)

