import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from config import *


class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset, mod):
        self.test_dataset = test_dataset
        self.mod = mod
        self.test_batch = next(iter(self.test_dataset))

    def on_epoch_end(self, epoch, logs=None):
        test_images, ground_truth_masks = self.test_batch
        predicted_masks = self.mod.predict(test_images) > 0.5
        self.plot_images(test_images, predicted_masks, ground_truth_masks, epoch)
        print('\n\n')

    def plot_images(self, images, predicted_masks, ground_truth_masks, epoch):
        images1, images2 = images
        num_images = min(5, images1.shape[0])

        plt.figure(figsize=(15, 5 * num_images))

        for i in range(num_images):
            plt.subplot(num_images, 4, 4 * i + 1)
            plt.imshow(images1[i])
            plt.title('Input Image')

            plt.subplot(num_images, 4, 4 * i + 2)
            plt.imshow(images2[i])
            plt.title('ELA Image')

            plt.subplot(num_images, 4, 4 * i + 3)
            plt.imshow(np.squeeze(predicted_masks[i]), cmap='gray')
            plt.title('Predicted Mask')

            plt.subplot(num_images, 4, 4 * i + 4)
            plt.imshow(np.squeeze(ground_truth_masks[i]), cmap='gray')
            plt.title('Ground Truth Mask')

        plt.tight_layout()
        plt.savefig(save_path + model_name + f'/epoch_{epoch+1}_images.png')
        plt.close()
