import os
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from glob import glob
import cv2
import io
from config import *

def calculate_ela(original_image, quality=ELA_QUALITY):
    with io.BytesIO() as output:
        original_image.save(output, format='JPEG', quality=quality)
        jpeg_data = output.getvalue()

    resaved_image = Image.open(io.BytesIO(jpeg_data))

    ela_image = ImageChops.difference(original_image, resaved_image)

    ela_image = ImageEnhance.Brightness(ela_image).enhance(6.0)

    return np.array(ela_image)

def load_data():
    images = list()
    masks = list()

    print('\n')
    for data_dir in data_dirs:
        image_path = path + data_dir + '/tp/*'
        mask_path = path + data_dir + '/gt/*'
        temp_images = sorted(glob(image_path))
        temp_masks = sorted(glob(mask_path))

        print("Image path:", image_path, 'Number:', len(temp_images))
        print("Mask path:", mask_path, 'Number:', len(temp_masks))
        images.extend(temp_images)
        masks.extend(temp_masks)

    print('\n')

    return images, masks

def read_images(path):
    img = Image.open(path)
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')

    x1 = np.array(img)
    x1 = cv2.resize(x1, IMAGE_SIZE)
    x1 = x1 / 255
    x1 = x1.astype(np.float32)

    x2 = calculate_ela(img)
    x2 = cv2.resize(x2, IMAGE_SIZE)
    x2 = x2 / 255
    x2 = x2.astype(np.float32)

    return x1, x2

def read_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, MASK_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def preprocess_batch(X_batch, y_batch):
    def f(X, y):
        X1_batch, X2_batch, y_batch = [], [], []
        for x, m in zip(X, y):
            x1, x2 = read_images(x)
            mask = read_mask(m)
            X1_batch.append(x1)
            X2_batch.append(x2)
            y_batch.append(mask)

        return X1_batch, X2_batch, y_batch

    X1_batch, X2_batch, y_batch = tf.numpy_function(f, [X_batch, y_batch], [tf.float32, tf.float32, tf.float32])
    X1_batch.set_shape((None, *IMAGE_SIZE, 3))
    X2_batch.set_shape((None, *IMAGE_SIZE, 3))
    y_batch.set_shape((None, *IMAGE_SIZE, 1))

    return (X1_batch, X2_batch), y_batch

def tf_dataset(X, y, batch=16):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=100_000, seed=SEED)
    dataset = dataset.batch(batch)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=8)
    dataset = dataset.prefetch(2 * 10)

    return dataset
