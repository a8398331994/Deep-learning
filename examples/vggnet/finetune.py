import os,sys
import numpy as np
import tensorflow as tf
import datetime
from VGGmodel import VggNetModel
from preprocessor import BatchPreprocessor

# The parsarse of tensorflow 
tf.app.flags.DEFINE_float('learning_rate',0.0001,'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob',0.5,'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs',10,'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes',26,'Number of classes')
tf.app.flags.DEFINE_integer('batch_size',128,'Batch size')
tf.app.flags.DEFINE_string('train_layers','fc8,fc7','Finetuning layers,seperated by commas')
tf.app.flags.DEFINE_string('multi_scale','','As preprocessing; scale the image randomly between 2 nimbers and crop randomly at network')
tf.app.flags.DEFINE_string('training_file','../data/val.txt','Training dataset file')
tf.app.flags.DEFINE_string('val_file','../data/val.txt','Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir','../training','Root directory to put the training log and weights')
tf.app.flags.DEFINE_integer('log_step',10,'Logging period in terms of iteration')

FLAGS=tf.app.flags.FLAGS

def main(_):
    # Create training directories
    now=datetime.datetime.now()
    print(now)
    train_dir_name=now.strftime('vggnet_%Y%m%d_%H%M%S')
    train_dir=os.path.join(FLAGS.tensorboard_root_dir,train_dir_name)
    checkpoint_dir=os.path.join(train_dir,'checkpoint')
    tensorboard_dir=os.path.join(train_dir,'tensorboard')
    tensorboard_train_dir=os.path.join(tensorboard_dir,'train')
    tensorboard_val_dir=os.path.join(tensorboard_dir,'val')

    # Create the dir if the dir does not exist
    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path=os.path.join(train_dir,'flags.txt')
    flags_file=open(flags_file_path,'w')
    flags_file.write('---{}---\n'.format(now))
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))

if __name__ == '__main__':
    tf.app.run()