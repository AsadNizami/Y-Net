import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.shared_dense_one = layers.Dense(channels//self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.shared_dense_two = layers.Dense(channels, kernel_initializer='he_normal', use_bias=True)

    def call(self, inputs):
        avg_pool = layers.GlobalAveragePooling2D()(inputs)    
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, max_pool.shape[1]))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)
        cbam_feature = tf.nn.sigmoid(avg_pool + max_pool)
        return layers.Multiply()([inputs, cbam_feature])

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(1, (self.kernel_size, self.kernel_size), padding='same', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(inputs)
        max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(inputs)
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        cbam_feature = self.conv2d(concat)
        cbam_feature = layers.Activation('sigmoid')(cbam_feature)
        return layers.Multiply()([inputs, cbam_feature])

class CBAM(layers.Layer):
    def __init__(self, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        ratio=8
        kernel_size=7
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        cbam_feature = self.channel_attention(inputs)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature
