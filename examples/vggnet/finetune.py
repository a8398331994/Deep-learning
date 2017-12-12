import os,sys
import numpy as np
import tensorflow as tf
import datetime
from VGGmodel import VggNetModel
from preprocessor import BatchPreprocessor

# The parsarse of tensorflow 
tf.app.flags.DEFINE_float('learning_rate',0.0001,'Learning rate for adam optimizer')

FLAGS=tf.app.flags.FLAGS

def main(_):
    print(FLAGS.learning_rate)
    pass

if __name__ == '__main__':
    tf.app.run()