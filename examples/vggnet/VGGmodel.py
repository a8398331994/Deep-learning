# This is ref about the :https://github.com/dgurkaynak/tensorflow-cnn-finetune/blob/master/vggnet/model.py#L15
import tensorflow as tf
import numpy as np

#use the vgg-D type
class VggNetModel(object):

    def __init__(self,num_classes=1000,dropout_keep_prob=0.5,conv_stddev=0.01):
        self.num_classes=num_classes
        self.dropout_keep_prob=dropout_keep_prob
        self.conv_stddev=conv_stddev

        self.weights={
            'wc1_1':tf.Variable(tf.random_normal([3,3,3,64],stddev=self.conv_stddev)),
            'wc1_2':tf.Variable(tf.random_normal([3,3,64,64],stddev=self.conv_stddev)),
            'wc2_1':tf.Variable(tf.random_normal([3,3,64,128],stddev=self.conv_stddev)),
            'wc2_2':tf.Variable(tf.random_normal([3,3,128,128],stddev=self.conv_stddev)),
            'wc3_1':tf.Variable(tf.random_normal([3,3,128,256],stddev=self.conv_stddev)),
            'wc3_2':tf.Variable(tf.random_normal([3,3,256,256],stddev=self.conv_stddev)),
            'wc3_3':tf.Variable(tf.random_normal([3,3,256,256],stddev=self.conv_stddev)),
            'wc4_1':tf.Variable(tf.random_normal([3,3,256,512],stddev=self.conv_stddev)),
            'wc4_2':tf.Variable(tf.random_normal([3,3,512,512],stddev=self.conv_stddev)),
            'wc4_3':tf.Variable(tf.random_normal([3,3,512,512],stddev=self.conv_stddev)),
            'wc5_1':tf.Variable(tf.random_normal([3,3,512,512],stddev=self.conv_stddev)),
            'wc5_2':tf.Variable(tf.random_normal([3,3,512,512],stddev=self.conv_stddev)),
            'wc5_3':tf.Variable(tf.random_normal([3,3,512,512],stddev=self.conv_stddev))
        }
        self.biases={
            'bc1_1':tf.Variable(tf.zeros([64])),
            'bc1_2':tf.Variable(tf.zeros([64])),
            'bc2_1':tf.Variable(tf.zeros([128])),
            'bc2_2':tf.Variable(tf.zeros([128])),
            'bc3_1':tf.Variable(tf.zeros([256])),
            'bc3_2':tf.Variable(tf.zeros([256])),
            'bc3_3':tf.Variable(tf.zeros([256])),
            'bc4_1':tf.Variable(tf.zeros([512])),
            'bc4_2':tf.Variable(tf.zeros([512])),
            'bc4_3':tf.Variable(tf.zeros([512])),
            'bc5_1':tf.Variable(tf.zeros([512])),
            'bc5_2':tf.Variable(tf.zeros([512])),
            'bc5_3':tf.Variable(tf.zeros([512]))
        }

    def inference(self,x,training=False):
        #conv1_1 size:64
        with tf.name_scope('conv1_1'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc1_1']
                variable_summaries(kernel)
            with tf.name_scope('biases'):
                biases=self.biases['bc1_1']
                variable_summaries(biases)
            conv=tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding='SAME')
            conv1_1=tf.nn.relu(tf.nn.bias_add(conv,biases))
        
        #conv1_2 
        with tf.name_scope('conv1_2'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc1_2']
            with tf.name_scope('biases'):
                biases=self.biases['bc1_2']
            conv=tf.nn.conv2d(conv1,kernel,strides=[1,1,1,1],padding='SAME')
            conv1_2=tf.nn.relu(tf.nn.bias_add(conv,biases))

        pool1=tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')

        #conv2_1 size:128
        with tf.name_scope('conv2_1'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc2_1']
            with tf.name_scope('biases'):
                biases=self.biases['bc2_1']
            conv=tf.nn.conv2d(pool1,kernel,strides=[1,1,1,1],padding='SAME')
            conv2_1=tf.nn.relu(tf.nn.bias_add(conv,biases))
        
        #conv2_2 
        with tf.name_scope('conv2_2'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc2_2']
            with tf.name_scope('biases'):
                biases=self.biases['bc2_2']
            conv=tf.nn.conv2d(conv2_1,kernel,strides=[1,1,1,1],padding='SAME')
            conv2_2=tf.nn.relu(tf.nn.bias_add(conv,biases))

        pool2=tf.nn.max_pool(conv2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')

        # conv3_1 size:256
        with tf.name_scope('conv3_1'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc3_1']
            with tf.name_scope('biases'):
                biases=self.biases['bc3_1']
            conv=tf.nn.conv2d(pool2,kernel,strides=[1,1,1,1],padding='SAME')
            conv3_1=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv3_2
        with tf.name_scope('conv3_2'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc3_2']
            with tf.name_scope('biases'):
                biases=self.biases['bc3_2']
            conv=tf.nn.conv2d(conv3_1,kernel,strides=[1,1,1,1],padding='SAME')
            conv3_2=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv3_3
        with tf.name_scope('conv3_3'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc3_3']
            with tf.name_scope('biases'):
                biases=self.biases['bc3_3']
            conv=tf.nn.conv2d(conv3_2,kernel,strides=[1,1,1,1],padding='SAME')
            conv3_3=tf.nn.relu(tf.nn.bias_add(conv,biases))

        pool3=tf.nn.max_pool(conv3_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')

        # conv4_1 size:512
        with tf.name_scope('conv4_1'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc4_1']
            with tf.name_scope('biases'):
                biases=self.biases['bc4_1']
            conv=tf.nn.conv2d(pool3,kernel,strides=[1,1,1,1],padding='SAME')
            conv4_1=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv4_2
        with tf.name_scope('conv4_2'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc4_2']
            with tf.name_scope('biases'):
                biases=self.biases['bc4_2']
            conv=tf.nn.conv2d(conv4_1,kernel,strides=[1,1,1,1],padding='SAME')
            conv4_2=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv4_3
        with tf.name_scope('conv4_3'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc4_3']
            with tf.name_scope('biases'):
                biases=self.biases['bc4_3']
            conv=tf.nn.conv2d(conv4_2,kernel,strides=[1,1,1,1],padding='SAME')
            conv4_3=tf.nn.relu(tf.nn.bias_add(conv,biases))

        pool4=tf.nn.max_pool(conv4_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')

        # conv5_1 size:512
        with tf.name_scope('conv5_1'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc5_1']
            with tf.name_scope('biases'):
                biases=self.biases['bc5_1']
            conv=tf.nn.conv2d(pool4,kernel,strides=[1,1,1,1],padding='SAME')
            conv5_1=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv5_2
        with tf.name_scope('conv5_2'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc5_2']            
            with tf.name_scope('biases'):
                biases=self.biases['bc5_2']               
            conv=tf.nn.conv2d(conv5_1,kernel,strides=[1,1,1,1],padding='SAME')
            conv5_2=tf.nn.relu(tf.nn.bias_add(conv,biases))

        # conv5_3 
        with tf.name_scope('conv5_3'):
            with tf.name_scope('weight'):
                kernel=self.weights['wc5_3']
                variable_summaries(kernel)
            with tf.name_scope('biases'):
                biases=self.biases['bc5_3']
                variable_summaries(biases)
            conv=tf.nn.conv2d(conv5_2,kernel,strides=[1,1,1,1],padding='SAME')
            conv5_3=tf.nn.relu(tf.nn.bias_add(conv,biases))

        pool5=tf.nn.max_pool(conv5_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5')

        # full connect 1
        with tf.name_scope('fc6'):
            shape=int(np.prob(pool5.get_shape()[1:]))
            print(shape)
            with tf.name_scope('weight'):
                fc6w=tf.Variable(tf.random_normal([shape,4096],stddev=self.stddev))
                variable_summaries(fc6w)
            with tf.name_scope('biases'):
                fc6b=tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=[4096]))
                variable_summaries(fc6b)
            pool5_flat=tf.reshape(pool5,[-1,shape])
            with tf.name_scope('W_b'):
                fc61=tf.nn.bias_add(tf.matmul(pool5_flat,fc6w),fc6b)
            fc6=tf.nn.relu(fc61)

            if training:
                fc6=tf.nn.dropout(fc6,self.dropout_keep_prob)

        # full connect 2
        with tf.name_scope('fc7'):
            with tf.name_scope('weight'):
                fc7w=tf.Variable(tf.random_normal([4096,4096],stddev=self.stddev))
            with tf.name_scope('biases'):
                fc7b=tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=[4096]))
            with tf.name_scope('W_b'):
                fc71=tf.nn.bias_add(tf.matmul(fc6,fc7w),fc7b)
            fc7=tf.nn.relu(fc71)

            if training:
                fc7=tf.nn.dropout(fc7,self.dropout_keep_prob)

        # full connect 3
        with tf.name_scope('fc8'):
            with tf.name_scope('weight'):
                fc8w=tf.Variable(tf.random_normal([4096,self.num_classes],stddev=self.stddev))
            with tf.name_scope('biases'):
                fc8b=tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=[self.num_classes]))
            with tf.name_scope('W_b'):
                self.score=tf.nn.bias_add(tf.matmul(fc7,fc8w),fc8b)

        return self.score

    def variable_summaries(var):# This function is to print the weights and biases value to tensorboard
        with tf.name_scope('summaries'):
            mean=tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            with tf.name_scope('stddev'):
                stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram',var)

    def loss(self,batch_x,batch_y=None):
        y_predict=self.inference(batch_x,training=True)
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=batch_y))
        return self.loss

    def optimize(self,learning_rate,train_layer=[]):
        # To get all the trainable variable
        var_list=[v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layer]
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss,var_list=var_list)

    def load_original_weights(self,session,skip_layers=[]):
        weights=np.load('vgg16_weights.npz')
        keys=sorted(weights.keys())

        for i,name in enumerate(keys):
            parts=name.spilt('_')
            layer='_'.join(part[:-1])
            print(len(keys),layer)
            # if layer in skip_layers:
            #     continue
            if layer == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope(layer, reuse=True):
                if parts[-1]=='W':
                    var=tf.get_variable('weight')
                    session.run(var.assign(weights[name]))
                elif parts[-1]=='b':
                    var=tf.get_variable('biases')
                    session.run(var.assign(weights[name]))

