import tensorflow as tf
import random
import numpy as np
from datetime import datetime

tf.keras.mixed_precision.set_global_policy('mixed_float16')
##
# Expected directory structure
# ...path/
#       d1/
#           tp/ images
#           gt/ masks
#       d2/ 
#           ...
#       d3/
#           ...
#       ...
##

path = '/media/ubuntu/New Volume1/dataset/train/'
data_dirs = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'dcas_1_2', 'd_cpy_mv', 'dfr']


save_path = './logs/'
model_name = 'ynet'  # A new directory will be created for this model

logdir = save_path + 'diag'

IMAGE_SIZE = (256, 256)
MASK_SIZE = IMAGE_SIZE
SEED = 7
ELA_QUALITY = 95
BATCH = 28

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

epochs = 60
INIT_LR = 3e-3

optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR)

SAVE_FLAG = True

# export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export LD_LIBRARY_PATH=${CUDNN_PATH}/lib
