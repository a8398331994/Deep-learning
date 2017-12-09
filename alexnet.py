import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tep/data/",one_hot=True)

#define the network Hyper parameters
learning_rate=0.001
training_iters=200000
batch_size=128
display_step=50

#define network parameters
n_input=784#img shape=28*28
n_classes=10
dropout=0.75

#define saver
ckpt_dir="./ckpt_alexnet"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# define placeholder
x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.float32)

# create the network model
def conv2d(name,x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)# activative function

# define the pooling function
def maxpool2d(name,x,k=2):
    # The ksize mean:the first and last parameters are mean batch_size and channels which 
    # we dont want to take the maximum over the multiple examples 
    # The second and third parameters are windows size over which you take the maximum 
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME",name=name)

#規範化操作
def norm(name,l_input,lsize=4):
    return tf.nn.lrn(l_input,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

# define all the network variables
weights={
    'wc1':tf.Variable(tf.random_normal([11,11,1,96])),#patch 11x11, in size 1(mean image of channel ),out size 96(mean output sizes depth is 96)
    'wc2':tf.Variable(tf.random_normal([5,5,96,256])),
    'wc3':tf.Variable(tf.random_normal([3,3,256,384])),
    'wc4':tf.Variable(tf.random_normal([3,3,384,384])),
    'wc5':tf.Variable(tf.random_normal([3,3,384,256])),
    'wd1':tf.Variable(tf.random_normal([2*2*256,1024])),
    'wd2':tf.Variable(tf.random_normal([1024,1024])),
    'out':tf.Variable(tf.random_normal([1024,10]))
}
biases={
    'bc1':tf.Variable(tf.random_normal([96])),
    'bc2':tf.Variable(tf.random_normal([256])),
    'bc3':tf.Variable(tf.random_normal([384])),
    'bc4':tf.Variable(tf.random_normal([384])),
    'bc5':tf.Variable(tf.random_normal([256])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'bd2':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# define all the net
def alex_net(x,weights,biases,dropout):
    # reshape input picture
    x=tf.reshape(x,shape=[-1,28,28,1])

    # first conv2d
    conv1=conv2d('conv1',x,weights['wc1'],biases['bc1'])
    # pooling
    pool1=maxpool2d('pool1',conv1,k=2)
    # 規範化
    norm1=norm('norm1',pool1,lsize=4)
    
    # second conv2d
    conv2=conv2d('conv2',norm1,weights['wc2'],biases['bc2'])
    # pooling
    pool2=maxpool2d('pool2',conv2,k=2)
    # 規範化
    norm2=norm('norm2',pool2,lsize=4)

    # third conv2d
    conv3=conv2d('conv3',norm2,weights['wc3'],biases['bc3'])
    # pooling
    pool3=maxpool2d('pool3',conv3,k=2)
    # 規範化
    norm3=norm('norm3',pool3,lsize=4)

    # 4th conv2d
    conv4=conv2d('conv4',norm3,weights['wc4'],biases['bc4'])
    # 5th conv2d
    conv5=conv2d('conv5',conv4,weights['wc5'],biases['bc5'])
    # pooling
    pool5=maxpool2d('pool5',conv5,k=2)
    print("pool5.shape:",pool5.get_shape())
    # 規範化
    norm5=norm('norm5',pool5,lsize=4)
    print("norm5.shape:",norm5.get_shape())
    

    # first full connect
    fc1=tf.reshape(norm5,[-1,weights['wd1'].get_shape().as_list()[0]])#[n_samples,4,4,256]->>[n_samples,4*4*256] that is become 1 dimension
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)
    # dropout
    fc1=tf.nn.dropout(fc1,dropout)
    print("fc1.shape:",fc1.get_shape())
    # second full connect
    fc2=tf.reshape(fc1,[-1,weights['wd2'].get_shape().as_list()[0]])#[n_samples,4,4,256]->>[n_samples,4*4*256] that is become 1 dimension
    fc2=tf.add(tf.matmul(fc2,weights['wd2']),biases['bd2'])
    fc2=tf.nn.relu(fc2)
    # dropout
    fc2=tf.nn.dropout(fc2,dropout)
    print("fc2.shape:",fc2.get_shape())
    # output layer
    out=tf.add(tf.matmul(fc2,weights['out']),biases['out'])
    print("out.shape:",out.get_shape())
    return out

# create model
pred=alex_net(x,weights,biases,keep_prob)
# define loss function and Optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# evalueate the function
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# counter variable
global_step=tf.Variable(0,name='global_step',trainable=False)
saver=tf.train.Saver()

# init variable 
init=tf.global_variables_initializer()
with tf.Session() as sess:
    # export graph
    writer = tf.summary.FileWriter('logs/',sess.graph)
    sess.run(init)
    start=global_step.eval()# To gain the value
    step=1
    # start training until train_iters be 200000
    while step* display_step<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
        if step%display_step==0:
            # calculate the loss and accuracy
            loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
            print("Iter"+str(step*display_step)+", Minibatch Loss="+"{:.6f}".format(loss)+", Training Accuracy="+"{:.5f}".format(acc))
            saver.save(sess,ckpt_dir+"/model.ckpt",global_step=global_step)
        step=step+1
        global_step.assign(step).eval()# update counter
    print("Optimization Finished")

    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.0}))