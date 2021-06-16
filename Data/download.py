import tensorflow as tf
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline

tf.random.set_seed(1)
np.random.seed(1)

path = 'CASIA-WebFace'

################ Creating Data ##########################
def parse_directory(dir_pos, dir_neg):
  pos_files = os.listdir(dir_pos)
  neg_files = os.listdir(dir_neg)

  pos_1 = os.path.join(dir_pos, pos_files[0])
  pos_2 = os.path.join(dir_pos, pos_files[1])
  pos_3 = os.path.join(dir_pos, pos_files[2])

  neg_1 = os.path.join(dir_neg, neg_files[0])
  neg_2 = os.path.join(dir_neg, neg_files[1])
  neg_3 = os.path.join(dir_neg, neg_files[2])

  pos_pair_1 = (pos_1, pos_2) # + +
  pos_pair_2 = (pos_1, pos_3) # + +
  pos_pair_3 = (pos_2, pos_3) # + +
  pos_pair_4 = (neg_1, neg_2) # - -
  pos_pair_5 = (neg_1, neg_3) # - -
  pos_pair_6 = (neg_2, neg_3) # - -
  
  neg_pair_1 = (pos_1, neg_1) # + -
  neg_pair_2 = (pos_1, neg_2) # + -
  neg_pair_3 = (pos_1, neg_3) # + -
  neg_pair_4 = (pos_2, neg_1) # + -
  neg_pair_5 = (pos_2, neg_2) # + -
  neg_pair_6 = (pos_2, neg_3) # + -
  neg_pair_7 = (pos_3, neg_1) # + -
  neg_pair_8 = (pos_3, neg_2) # + -
  neg_pair_9 = (pos_3, neg_3) # + -

  return {'pos':[pos_pair_1, pos_pair_2, pos_pair_3, 
                 pos_pair_4, pos_pair_5, pos_pair_6], 
          'neg':[neg_pair_1, neg_pair_2, neg_pair_3, 
                 neg_pair_4, neg_pair_5, neg_pair_6, 
                 neg_pair_7, neg_pair_8, neg_pair_9]}

def create_pairs(list_ds, total_file=5287):
  whole_pairs ={'neg': [], 'pos': []}
  for i in list_ds.take(5287):
    # This file has 2 image only
    if i.numpy()[0].decode('utf-8') == 'CASIA-WebFace/2986449':
      continue
    pairs = parse_directory(i.numpy()[0].decode('utf-8'), i.numpy()[1].decode('utf-8'))
    whole_pairs['neg'].append(pairs['neg'])
    whole_pairs['pos'].append(pairs['pos'])
  return whole_pairs
########################################################################

########## Data to DataFrame ######################
def get_pairs(whole_pairs, tag):
  image_1 = []
  image_2 = []
  for pairs in whole_pairs[tag]:
    for pair in pairs:
      image_1.append(pair[0])
      image_2.append(pair[1])
  return image_1, image_2

def create_df(whole_pairs):
  # calling pairs for each image, they are lists
  pos_img_1, pos_img_2 = get_pairs(whole_pairs, 'pos')
  neg_img_1, neg_img_2 = get_pairs(whole_pairs, 'neg')

  df_pos = pd.DataFrame(data={'image1':pos_img_1, 'image2':pos_img_2, 'similarity':1.0})
  df_neg = pd.DataFrame(data={'image1':neg_img_1, 'image2':neg_img_2, 'similarity':0.0})
  df_total = df_pos.append(df_neg)
  # Shuffling
  df_total = df_total.sample(frac=1).reset_index(drop=True)
  return df_total
def save_df_to_csv(df, name):
  df.to_csv(name, index=False)
  print('DataFrame successfully saved.')
###################################################

########### Visualiziatin #########################
def parse_image(filename, filename2):

  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [250, 250])

  image2 = tf.io.read_file(filename2)
  image2 = tf.image.decode_jpeg(image2)
  image2 = tf.image.convert_image_dtype(image2, tf.float32)
  image2 = tf.image.resize(image2, [250, 250])
  return image, image2

def show(image, image2, label):
  plt.figure()
  _, axarr = plt.subplots(1,2)
  axarr[0].imshow(image)
  axarr[1].imshow(image2)
  # plt.imshow(image)
  plt.title(label, loc='left')
  axarr[0].axis('off')
  axarr[1].axis('off')
##################################################

faces_root = pathlib.Path(path)
list_ds = tf.data.Dataset.list_files(str(faces_root/'*')).batch(2)
whole_pairs = create_pairs(list_ds)
df = create_df(whole_pairs)
save_df_to_csv(df, 'CASIA_WebFace_pretrained.csv')