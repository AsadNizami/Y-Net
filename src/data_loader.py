from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from glob import glob
import cv2
import io
from config import *


def calculate_ela(image, quality=ELA_QUALITY):
    with io.BytesIO() as buf:
        image.save(buf, format='JPEG', quality=quality)
        jpeg_data = buf.getvalue()
    resaved = Image.open(io.BytesIO(jpeg_data))
    ela = ImageChops.difference(image, resaved)
    ela = ImageEnhance.Brightness(ela).enhance(6.0)
    return np.array(ela)


def load_data():
    images, masks = [], []
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


def load_sample(img_path, mask_path):
    """Decode + ELA + resize. No augmentation — that runs in-graph after caching."""
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    mask = Image.open(mask_path).convert('L')

    x2 = calculate_ela(img)
    x2 = cv2.resize(x2, IMAGE_SIZE) / 255.0
    x2 = x2.astype(np.float32)

    x1 = cv2.resize(np.array(img), IMAGE_SIZE) / 255.0
    x1 = x1.astype(np.float32)

    mask_arr = np.array(mask, dtype=np.float32)
    mask_arr = cv2.resize(mask_arr, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST) / 255.0
    mask_arr = mask_arr[:, :, np.newaxis]

    return x1, x2, mask_arr


def load_one(img_path, mask_path):
    def f(p, m):
        p = p.decode() if isinstance(p, bytes) else p
        m = m.decode() if isinstance(m, bytes) else m
        return load_sample(p, m)

    x1, x2, mask = tf.numpy_function(
        f, [img_path, mask_path], [tf.float32, tf.float32, tf.float32]
    )
    x1.set_shape((*IMAGE_SIZE, 3))
    x2.set_shape((*IMAGE_SIZE, 3))
    mask.set_shape((*IMAGE_SIZE, 1))
    return (x1, x2), mask


def tf_augment(inputs, mask):
    """In-graph augmentation: flips, 90° rotations, color jitter on RGB only."""
    x1, x2 = inputs

    # Geometric: applied identically to RGB, ELA, and mask via channel concat
    combined = tf.concat([x1, x2, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k=k)

    x1 = combined[..., 0:3]
    x2 = combined[..., 3:6]
    mask = combined[..., 6:7]

    # Color jitter: RGB only — preserves ELA artifact pattern
    x1 = tf.image.random_brightness(x1, max_delta=0.2)
    x1 = tf.image.random_contrast(x1, lower=0.8, upper=1.2)
    x1 = tf.image.random_saturation(x1, lower=0.8, upper=1.2)
    x1 = tf.clip_by_value(x1, 0.0, 1.0)

    return (x1, x2), mask


def tf_dataset(X, y, batch=16, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        dataset = dataset.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_one, num_parallel_calls=4)
    if augment:
        dataset = dataset.map(tf_augment, num_parallel_calls=4)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset
