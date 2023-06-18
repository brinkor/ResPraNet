'''
MIT License

Copyright (c) 2020 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import tensorflow as tf
from tensorflow.keras import layers
from model.conv_module import ConvModule
from model.res_block import ResModule


class RFB(tf.keras.layers.Layer):
    def __init__(self, filters: int, name: str):
        super(RFB, self).__init__(name=name)
        self.filters = filters

        self.branch_1 = tf.keras.Sequential([
            ResModule(filters=self.filters, kernelsize=(1, 1))
        ])
        self.branch_2 = tf.keras.Sequential([
            ResModule(filters=self.filters, kernelsize=(1, 1)),
            ResModule(filters=self.filters, kernelsize=(1, 3)),
            ResModule(filters=self.filters, kernelsize=(3, 1)),
            ResModule(filters=self.filters, kernelsize=(
                3, 3), dilation_rate=(3, 3))
        ])

        self.branch_3 = tf.keras.Sequential([
            ResModule(filters=self.filters, kernelsize=(1, 1)),
            ResModule(filters=self.filters, kernelsize=(1, 5)),
            ResModule(filters=self.filters, kernelsize=(5, 1)),
            ResModule(filters=self.filters, kernelsize=(
                3, 3), dilation_rate=(5, 5))
        ])

        self.branch_4 = tf.keras.Sequential([
            ResModule(filters=self.filters, kernelsize=(1, 1)),
            ResModule(filters=self.filters, kernelsize=(1, 7)),
            ResModule(filters=self.filters, kernelsize=(7, 1)),
            ResModule(filters=self.filters, kernelsize=(
                3, 3), dilation_rate=(7, 7))
        ])

        self.concate_branch = ResModule(
            filters=self.filters, kernelsize=(1, 1))

        self.shortcut_branch = ResModule(
            filters=self.filters, kernelsize=(1, 1))

        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.nn.relu(inputs)
        x1 = self.branch_1(inputs)
        x2 = self.branch_2(inputs)
        x3 = self.branch_3(inputs)
        x4 = self.branch_4(inputs)
        # x_res = self.shortcut_branch(inputs)

        x_con = tf.concat([x1, x2, x3, x4], axis=-1)

        x_concat_conv = self.concate_branch(x_con)

        # x = self.relu(x_concat_conv + x_res)
        x = self.relu(x_concat_conv)

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"filters": self.filters})
        return config

    @classmethod
    def from_config(cls, config):
        return super().from_config(config)
    
    
    def resblock(self, x, kernelsize, filters):
        fx = layers.Conv2D(filters, kernelsize, activation='relu', padding='same')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Conv2D(filters, kernelsize, padding='same')(fx)
        out = layers.Add()([x,fx])
        out = layers.ReLU()(out)
        out = layers.BatchNormalization()(out)
        return out


# test the module
if __name__ == "__main__":
    rfb = RFB(32, "RFB_test")
    # first call to the `rfb` will create weights
    y = rfb(tf.ones(shape=(8, 7, 7, 2048)))

    print("weights:", len(rfb.weights))
    print("trainable weights:", len(rfb.trainable_weights))
    print("config:", rfb.get_config())
    print(f"Y: {y.shape}")
