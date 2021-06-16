
import tensorflow as tf
from .slim_module import SlimModule
from .config import Config

class SlimCNN(tf.keras.Model):
    
    def __init__(self, 
                slim_module_filters, 
                conv_filters_size, 
                conv_filters, 
                conv_strides, 
                max_pool_filter, 
                max_pool_stride, 
                activation='relu', 
                regularizer=1e-4,
                n_class=40):
        
        super(SlimCNN, self).__init__()

        # self.input = tf.keras.layers.Input(shape=(178, 218, 3))
        self.conv = tf.keras.layers.Conv2D(conv_filters_size, conv_filters, strides=conv_strides, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        
        self.slim_module_1 = SlimModule(slim_module_filters['layer1'], l2_regularizer=regularizer)
        self.slim_module_2 = SlimModule(slim_module_filters['layer2'], l2_regularizer=regularizer)
        self.slim_module_3 = SlimModule(slim_module_filters['layer3'], l2_regularizer=regularizer)
        self.slim_module_4 = SlimModule(slim_module_filters['layer4'], l2_regularizer=regularizer)
        
        self.max_pool = tf.keras.layers.MaxPooling2D(max_pool_filter, max_pool_stride) 
        
        self.act = tf.keras.layers.Activation(activation)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(n_class, activation='sigmoid')
    
    def call(self, input_tensor, mask=None):
        # conv1 + MaxPool
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        # Slim module 1 + MaxPool
        x = self.slim_module_1(x)
        x = self.max_pool(x)

        # Slim module 2 + MaxPool
        x = self.slim_module_2(x)
        x = self.max_pool(x)

        # Slim module 3 + MaxPool
        x = self.slim_module_3(x)
        x = self.max_pool(x)

        # Slim module 4 + MaxPool
        x = self.slim_module_4(x)
        x = self.max_pool(x)
        
        # Global Average Pooling + FC
        return self.classifier(self.global_pool(x))
    
    def my_compile(self, model, dynamic_mode=False):
        ''' Dynamic mode is prepared for only build in function 
        uses. Other wise this function helps to work clearly for model 
        summary function. Also after load weights you should use dynamic_mode=True
        mode.
        '''
        input = tf.keras.layers.Input(shape=Config.input_size)
        model.build(input_shape=(None, 178, 218, 3))
        model.call(input)
        if dynamic_mode:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=Config.lr), 
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                        metrics=[tf.keras.metrics.BinaryAccuracy()])
            return model
        return model