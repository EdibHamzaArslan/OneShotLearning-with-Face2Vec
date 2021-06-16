import os
import numpy as np 
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    
    ''' Generates data for keras
    Sources:
    https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''

    def __init__(self, img1_list, img2_list, features, batch_size=64, dim=(178, 218), n_channels=3, shuffle=True):
        
        self.img1_list = img1_list 
        self.img2_list = img2_list
        self.features = features
        
        self.batch_size = batch_size
        self.dim = dim
        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        if self.batch_size > self.img1_list.shape[0]:
            print("Batch size is greater than data size!!")
            return -1
        return int(np.floor(self.img1_list.shape[0] / self.batch_size))

    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        image1_file_list = [self.img1_list[k] for k in indexes]
        image2_file_list = [self.img2_list[k] for k in indexes]

        X1, X2 = self.__generate_X(image1_file_list, image2_file_list)
        y = np.array([self.features[k] for k in indexes]).astype(np.float32)

        return (X1, X2), y

    def __generate_X(self, image1_file_list, image2_file_list):
        
        image1s_x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        image2s_x = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        
        for index in range(len(image1_file_list)):
            img = tf.keras.preprocessing.image.load_img(image1_file_list[index], target_size=self.dim)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img.astype("float32") / 255
            image1s_x[index] = img
        
        for index in range(len(image2_file_list)):
            img = tf.keras.preprocessing.image.load_img(image2_file_list[index], target_size=self.dim)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img.astype("float32") / 255
            image2s_x[index] = img
        
        return image1s_x, image2s_x


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.seed(2)
            np.random.shuffle(self.indexes)
    
    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

