import tensorflow as tf
from keras import layers


class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_one = layers.Dense(channels // self.ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.dense_two = layers.Dense(channels, kernel_initializer='he_normal', use_bias=True)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_pool = self.dense_two(self.dense_one(avg_pool))
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool = self.dense_two(self.dense_one(max_pool))
        return inputs * tf.sigmoid(avg_pool + max_pool)

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio})
        return config


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, self.kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return inputs * self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({'kernel_size': self.kernel_size})
        return config


class CBAM(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        return self.spatial_attention(x)

    def get_config(self):
        config = super().get_config()
        config.update({'ratio': self.ratio, 'kernel_size': self.kernel_size})
        return config
