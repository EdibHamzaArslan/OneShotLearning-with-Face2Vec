
import tensorflow as tf



class SSE(tf.keras.layers.Layer):
    
    def __init__(self, squeeze_filters, expand_1_filters, expand_3_filters, activation='relu'):
        super(SSE, self).__init__()
        self.squeeze = tf.keras.layers.Conv2D(squeeze_filters, (1, 1), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.expand_1 = tf.keras.layers.Conv2D(expand_1_filters, (1, 1), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.expand_3 = tf.keras.layers.SeparableConv2D(expand_3_filters, (3, 3), padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.concatenate = tf.keras.layers.Concatenate()
        
        self.act = tf.keras.layers.Activation(activation)

        
    def call(self, inputs):
        x = inputs
        
        x = self.squeeze(x)
        x = self.bn1(x)
        x = self.act(x)

        hidden1 = self.expand_1(x)
        hidden1 = self.bn2(hidden1)
        hidden1 = self.act(hidden1)

        hidden2 = self.expand_3(x)
        hidden2 = self.bn3(hidden2)
        hidden2 = self.act(hidden2)

        x = self.concatenate([hidden1, hidden2])
        return x
        