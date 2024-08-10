import tensorflow as tf
import random
import numpy as np
from datetime import datetime

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

path = '/media/asad/Windows-SSD/Users/ASAD/Documents/Dissertation/bootstrap/train/'
data_dirs = ['dcas_1_2']


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
