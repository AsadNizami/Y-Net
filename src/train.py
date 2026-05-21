import os
import math
import pickle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers.schedules import CosineDecay
from config import *
from custom_callback import *
from custom_layer import CBAM
from data_loader import *
from metrics import *
from model import *
from plot import *

print('=============================================================================')
print('Path:', path)
print('Batch:', BATCH)
print('Epochs:', epochs)
print('ELA:', ELA_QUALITY)
print('Learning rate:', INIT_LR)
print('=============================================================================')

images, masks = load_data()

# Shuffle at path level for a clean train/val split
n = len(images)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

train_n = int(0.75 * n)
train_images, train_masks = list(images[:train_n]), list(masks[:train_n])
val_images, val_masks = list(images[train_n:]), list(masks[train_n:])

train = tf_dataset(train_images, train_masks, batch=BATCH, augment=True)
val = tf_dataset(val_images, val_masks, batch=BATCH, augment=False)

print('=============================================================================')
print(f'Total images: {n}  |  Train: {train_n}  |  Val: {n - train_n}')
print('=============================================================================')

# Cosine decay: smoothly anneals LR to near-zero over the full training run
steps_per_epoch = math.ceil(train_n / BATCH)
total_steps = steps_per_epoch * epochs
lr_schedule = CosineDecay(INIT_LR, total_steps, alpha=1e-6)
optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

model = YNet()

custom_objects = {
    'CBAM': CBAM,
    'dice_loss': dice_loss,
    'dice_coefficient': dice_coefficient,
    'iou': iou,
    'accuracy': accuracy,
    'tversky_bce_loss': tversky_bce_loss,
    'focal_tversky_bce_loss': focal_tversky_bce_loss,
    'weighted_dice_bce_loss': weighted_dice_bce_loss,
    'precision': precision,
    'recall': recall,
}

# model = tf.keras.models.load_model('./logs/ablation_wo_cbam/epoch50_14.keras', custom_objects=custom_objects)
model.summary()

os.makedirs(save_path + model_name, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=save_path + model_name + '/epoch_{epoch:02d}.keras',
        monitor='val_loss', verbose=2, save_best_only=True, mode='min'
    ),
    ImageCallback(test_dataset=val, mod=model, every=5),
]

print('\nCallback created!\n')

model.compile(optimizer=optimizer, loss=focal_tversky_bce_loss, metrics=[dice_coefficient, iou, accuracy], jit_compile=True)  # type: ignore[arg-type]

print('\nModel compiled\n')

res = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)
plot_results(res)

if SAVE_FLAG:
    dictionary = res.history
    with open(save_path + model_name + '/history.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    add_row('results.csv', dictionary.copy())

model.save(save_path + model_name + '/final.keras')
