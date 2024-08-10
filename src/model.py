from tensorflow.keras.models import Model
from tensorflow.keras import layers
from config import *
from custom_layer import CBAM


def conv_block(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = CBAM()(x) 

    return x

def upconv_block(x, filters, kernel_size=3):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = CBAM()(x) 

    return x

def YNet(input_shape1=(*IMAGE_SIZE, 3), input_shape2=(*IMAGE_SIZE, 3)):
    inputs1 = layers.Input(input_shape1)
    
    conv0_s1 = conv_block(inputs1, 32)
    pool0_s1 = layers.MaxPooling2D(pool_size=(2, 2))(conv0_s1)

    conv1_s1 = conv_block(pool0_s1, 64)
    pool1_s1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1_s1)
    
    conv2_s1 = conv_block(pool1_s1, 128)
    pool2_s1 = layers.MaxPooling2D(pool_size=(2, 2))(conv2_s1)
    
    conv3_s1 = conv_block(pool2_s1, 256, kernel_size=5)
    pool3_s1 = layers.MaxPooling2D(pool_size=(2, 2))(conv3_s1)

    conv4_s1 = conv_block(pool3_s1, 256, kernel_size=5)
    pool4_s1 = layers.MaxPooling2D(pool_size=(2, 2))(conv4_s1)
    
    inputs2 = layers.Input(input_shape2)
    
    conv0_s2 = conv_block(inputs2, 32)
    pool0_s2 = layers.MaxPooling2D(pool_size=(2, 2))(conv0_s2)

    conv1_s2 = conv_block(pool0_s2, 64)
    pool1_s2 = layers.MaxPooling2D(pool_size=(2, 2))(conv1_s2)
    
    conv2_s2 = conv_block(pool1_s2, 128)
    pool2_s2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2_s2)
    
    conv3_s2 = conv_block(pool2_s2, 256, kernel_size=5)
    pool3_s2 = layers.MaxPooling2D(pool_size=(2, 2))(conv3_s2)

    conv4_s2 = conv_block(pool3_s2, 256, kernel_size=5) #
    pool4_s2 = layers.MaxPooling2D(pool_size=(2, 2))(conv4_s2) #
    
    merge = layers.Concatenate()([pool4_s1, pool4_s2]) 

    up4 = upconv_block(merge, 512, kernel_size=5)
    concat4 = layers.concatenate([up4, conv4_s1, conv4_s2], axis=-1)
    conv4 = conv_block(concat4, 512)
    conv4 = conv_block(conv4, 512)

    up3 = upconv_block(conv4, 256, kernel_size=5)
    concat3 = layers.concatenate([up3, conv3_s1, conv3_s2], axis=-1)
    conv3 = conv_block(concat3, 256)
    conv3 = conv_block(conv3, 256)

    up2 = upconv_block(conv3, 128)
    concat2 = layers.concatenate([up2, conv2_s1, conv2_s2], axis=-1)
    conv2 = conv_block(concat2, 128)
    conv2 = conv_block(conv2, 128)

    up1 = upconv_block(conv2, 64)
    concat1 = layers.concatenate([up1, conv1_s1, conv1_s2], axis=-1)
    conv1 = conv_block(concat1, 64)
    conv1 = conv_block(conv1, 64)

    up0 = upconv_block(conv1, 32)
    concat0 = layers.concatenate([up0, conv0_s1, conv0_s2], axis=-1)

    output = layers.Conv2D(1, (1, 1), activation='sigmoid')(concat0)
    
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model
