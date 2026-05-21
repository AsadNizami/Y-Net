from keras.models import Model
from keras import layers
from config import *
from custom_layer import CBAM


def conv_block(x, filters, kernel_size=3, stride=1, dropout=0.0):
    """Residual conv block: Conv → BN → ReLU → CBAM → (+shortcut) → ReLU → SpatialDropout."""
    shortcut = x
    in_channels = x.shape[-1]

    h = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = CBAM()(h)

    if in_channels != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    h = layers.Add()([h, shortcut])
    h = layers.ReLU()(h)

    if dropout > 0:
        h = layers.SpatialDropout2D(dropout)(h)

    return h


def upconv_block(x, filters, kernel_size=3):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = CBAM()(x)
    return x


def YNet(input_shape1=(*IMAGE_SIZE, 3), input_shape2=(*IMAGE_SIZE, 3)):
    # ----- RGB encoder: 256→128→64→32→16 -----
    inputs1 = layers.Input(input_shape1)

    conv0_s1 = conv_block(inputs1, 32)
    pool0_s1 = layers.MaxPooling2D()(conv0_s1)

    conv1_s1 = conv_block(pool0_s1, 64)
    pool1_s1 = layers.MaxPooling2D()(conv1_s1)

    conv2_s1 = conv_block(pool1_s1, 128)
    pool2_s1 = layers.MaxPooling2D()(conv2_s1)

    conv3_s1 = conv_block(pool2_s1, 256, kernel_size=5)
    pool3_s1 = layers.MaxPooling2D()(conv3_s1)

    # ----- ELA encoder: 256→128→64→32→16 -----
    inputs2 = layers.Input(input_shape2)

    conv0_s2 = conv_block(inputs2, 32)
    pool0_s2 = layers.MaxPooling2D()(conv0_s2)

    conv1_s2 = conv_block(pool0_s2, 64)
    pool1_s2 = layers.MaxPooling2D()(conv1_s2)

    conv2_s2 = conv_block(pool1_s2, 128)
    pool2_s2 = layers.MaxPooling2D()(conv2_s2)

    conv3_s2 = conv_block(pool2_s2, 256, kernel_size=5)
    pool3_s2 = layers.MaxPooling2D()(conv3_s2)

    # ----- Bottleneck at 16×16 -----
    merge = layers.Concatenate()([pool3_s1, pool3_s2])
    merge = conv_block(merge, 512, dropout=0.3)

    # ----- Decoder: 16→32→64→128→256 -----
    up3 = upconv_block(merge, 256, kernel_size=5)
    concat3 = layers.Concatenate()([up3, conv3_s1, conv3_s2])
    conv3 = conv_block(concat3, 256, dropout=0.3)
    conv3 = conv_block(conv3, 256, dropout=0.3)

    up2 = upconv_block(conv3, 128)
    concat2 = layers.Concatenate()([up2, conv2_s1, conv2_s2])
    conv2 = conv_block(concat2, 128, dropout=0.2)
    conv2 = conv_block(conv2, 128, dropout=0.2)

    up1 = upconv_block(conv2, 64)
    concat1 = layers.Concatenate()([up1, conv1_s1, conv1_s2])
    conv1 = conv_block(concat1, 64, dropout=0.1)
    conv1 = conv_block(conv1, 64, dropout=0.1)

    up0 = upconv_block(conv1, 32)
    concat0 = layers.Concatenate()([up0, conv0_s1, conv0_s2])
    concat0 = conv_block(concat0, 32)

    output = layers.Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(concat0)

    return Model(inputs=[inputs1, inputs2], outputs=output)
