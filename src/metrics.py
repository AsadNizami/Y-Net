import tensorflow as tf
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TruePositives, TrueNegatives
import tensorflow.keras.backend as K

TP = TruePositives()
TN = TrueNegatives()
FP = FalsePositives()
FN = FalseNegatives()

def calculate_metrics(y_true, y_pred):
    TP.update_state(y_true, y_pred)
    tp = TP.result()
    TP.reset_state()
    
    TN.update_state(y_true, y_pred)
    tn = TN.result()
    TN.reset_state()

    FP.update_state(y_true, y_pred)
    fp = FP.result()
    FP.reset_state()
    
    FN.update_state(y_true, y_pred)
    fn = FN.result()
    FN.reset_state()
    
    return tp, tn, fp, fn

def dice_coefficient(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    dice = 2 * prec * recal / (prec + recal)
    return dice

def soft_dice_coefficient(y_true, y_pred, smooth=1e-6):
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1-soft_dice_coefficient(y_true, y_pred)

def weighted_dice_bce_loss(y_true, y_pred):
    weight = 0.85

    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    loss = weight * dice + (1-weight) * bce
    
    return loss

def iou(y_true, y_pred):
    TP, TN, FP, FN = calculate_metrics(y_true, y_pred)
    iou = TP / (TP + FP + FN)

    return iou

def iou_loss(y_true, y_pred):
    return 1 - iou(y_true, y_pred)

def accuracy(y_true, y_pred):
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, tf.round(y_pred)), dtype=tf.float32))
    total_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)), dtype=tf.float32)
    accuracy = correct_pixels / total_pixels
    return accuracy

def precision(y_true, y_pred):
    TP, TN, FP, FN = calculate_metrics(y_true, y_pred)
    prec = (TP + K.epsilon()) / (TP + FP + K.epsilon())
    return prec
    
def recall(y_true, y_pred):
    TP, TN, FP, FN = calculate_metrics(y_true, y_pred)
    recal = TP / (TP + FN)
    return recal
