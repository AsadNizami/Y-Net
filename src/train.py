import os
import sys
from config import *
from custom_callback import *
from data_loader import *
from metrics import *
from model import *
import pickle
from plot import *
from custom_layer import CBAM
from metrics import *

print('=============================================================================')
print('Path:', path)
print('Batch:', BATCH)
print('Epochs:', epochs)
print('ELA:', ELA_QUALITY)
print('Learning rate:', INIT_LR)
print('=============================================================================')

images, masks = load_data()
dataset = tf_dataset(images, masks, batch=BATCH)

print('=============================================================================')
print('Number of images:', len(images))
print('=============================================================================')

DATASET_SIZE = len(dataset)
train_size = int(0.75 * DATASET_SIZE)
val_size = int(0.25 * DATASET_SIZE)

train = dataset.take(train_size)
val = dataset.skip(train_size)

# model = YNet()

custom_objects = {
    'CBAM': CBAM,
    'dice_loss': dice_loss,
    'dice_coefficient': dice_coefficient,
    'iou': iou,
    'accuracy': accuracy,
    'weighted_dice_bce_loss': weighted_dice_bce_loss,
    'precision': precision,
    'recall': recall
}

model = tf.keras.models.load_model('./logs/ablation_wo_cbam/epoch50_14.keras', custom_objects=custom_objects)
model.summary()

print('\nPath:', path)

if not os.path.exists(save_path + model_name):
    os.makedirs(save_path + model_name)
# else:
#     print('Directory already exists.')
#     sys.exit(0)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=save_path + model_name + '/epoch15_{epoch:02d}.keras', monitor='val_loss', verbose=2, save_best_only=True, mode='min'),
    ImageCallback(test_dataset=val, mod=model),
]

print('\nCallback created!\n')

model.compile(optimizer=optimizer, loss=weighted_dice_bce_loss, metrics=[dice_coefficient, iou, accuracy], jit_compile=True)

print('\nModel compiled\n')

res = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)
plot_results(res)

if SAVE_FLAG:
    dictionary = res.history
    with open(save_path + model_name + '/history.pkl', 'wb') as f:
        pickle.dump(dictionary, f)

    add_row('results.csv', dictionary.copy())

model.save(save_path + model_name + '/final.keras')
