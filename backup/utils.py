import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def create_directory(directory):
    for i in range(len(directory.split('/'))):
        if directory.split('/')[i] != '':
            sub_dic ='/'.join(directory.split('/')[:(i+1)])
            if not os.path.exists(sub_dic):
                os.makedirs(sub_dic)

