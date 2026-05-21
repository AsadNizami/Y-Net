import tensorflow as tf
from keras import backend as K


def _prepare(y_true, y_pred):
    y_pred = tf.cast(tf.round(tf.cast(y_pred, tf.float32)), tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    return y_true, y_pred


def precision(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    return (tp + K.epsilon()) / (tp + fp + K.epsilon())


def recall(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return (tp + K.epsilon()) / (tp + fn + K.epsilon())


def dice_coefficient(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    return 2.0 * prec * recal / (prec + recal + K.epsilon())


def soft_dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - soft_dice_coefficient(y_true, y_pred)


def weighted_dice_bce_loss(y_true, y_pred, weight=0.85):
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, tf.cast(y_pred, tf.float32))
    return weight * dice + (1.0 - weight) * bce


def tversky_index(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """alpha penalises FP, beta penalises FN. beta > alpha improves recall on small forged regions."""
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def tversky_loss(y_true, y_pred):
    return 1.0 - tversky_index(y_true, y_pred)


def tversky_bce_loss(y_true, y_pred, weight=0.85):
    tversky = tversky_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, tf.cast(y_pred, tf.float32))
    return weight * tversky + (1.0 - weight) * bce


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    """gamma < 1 focuses gradient on hard-to-segment pixels (boundaries)."""
    return tf.pow(tversky_loss(y_true, y_pred), gamma)


def focal_tversky_bce_loss(y_true, y_pred, weight=0.85):
    ftl = focal_tversky_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, tf.cast(y_pred, tf.float32))
    return weight * ftl + (1.0 - weight) * bce


def iou(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    return tp / (tp + fp + fn + K.epsilon())


def iou_loss(y_true, y_pred):
    return 1.0 - iou(y_true, y_pred)


def accuracy(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total = tf.cast(tf.size(y_true), tf.float32)
    return correct / total
