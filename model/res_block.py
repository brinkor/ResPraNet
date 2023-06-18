import tensorflow as tf
from tensorflow.keras import layers


class ResModule(tf.keras.layers.Layer):
    def __init__(
        self, filters: int,
        kernelsize: tuple,
        strides: tuple = (1, 1),
        padding: str = "same",
        dilation_rate: tuple = (1, 1),
    ):
        super(ResModule, self).__init__()
        self.filters = filters
        self.kernelsize = kernelsize
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate

        self.conv1 = layers.Conv2D(self.filters, self.kernelsize, dilation_rate=self.dilation_rate, strides=self.strides, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(self.filters, self.kernelsize, dilation_rate=self.dilation_rate, strides=self.strides, padding='same')
        self.add = layers.Add()
        self.relu = layers.ReLU()
        self.bn2 = layers.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(x)
        fx = self.bn1(x, training=training)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        out = self.add([x, fx])
        out = self.relu(out)
        out = self.bn2(out, training=training)

        return out

    def get_config(self) -> dict:
        config = super(ResModule, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernelsize,
            "strides": self.strides,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)