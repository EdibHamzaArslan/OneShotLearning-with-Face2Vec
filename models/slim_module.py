

import tensorflow as tf
from .sse_block import SSE


class SlimModule(tf.keras.Model):
    def __init__(self, sse_filters, activation='relu', l2_regularizer=1e-4):
        super().__init__()
        self.sse1 = SSE(sse_filters['squeeze'],
                        sse_filters['expand_1'],
                        sse_filters['expand_3'],
                        activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.skip_connection = tf.keras.layers.Conv2D(sse_filters['expand_1'] * 2,
                                                    (1, 1),
                                                    padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.sse2 = SSE(sse_filters['squeeze'],
                        sse_filters['expand_1'],
                        sse_filters['expand_3'],
                        activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.seperable_conv2d = tf.keras.layers.SeparableConv2D(filters=sse_filters['dw_seperable'],
                                                                kernel_size=(3, 3),
                                                                padding='same',
                                                                kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer))
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        
        x = self.sse1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        skip_connection = self.skip_connection(inputs)
        skip_connection = self.bn2(skip_connection)
        
        x = self.add([skip_connection, x])
        x = self.act(x)

        x = self.sse2(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.seperable_conv2d(x)
        x = self.bn4(x)
        x = self.act(x)

        return x

