import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data')
# define hyper parameter
learning_rate=0.01
training_epochs=20 # 訓練的倫數
batch_size=256
display_step=1 # 每隔多少輪顯示一次訓練結果

examples_to_show=10 # 顯示10個結果

# Network parameters
n_hidden_1=256 # 第一個隱藏層神經原各數，也是特徵質各數
n_hidden_2=128
n_input=784

# 因為這昰無監督式學習所以不用輸入labels
X=tf.placeholder("float",[None,n_input])

weights={
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}
biases={
    'encoder_h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_input]))
}

# define the tensor of summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)

# define encoder function
def encoder(x):
    # Encoder Hidden layer with sigmoid activation
    with tf.name_scope('layer1_encoder'):
        with tf.name_scope('weights'):
            variable_summaries(weights['encoder_h1'])
        with tf.name_scope('biases'):
            variable_summaries(biases['encoder_h1'])
        layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_h1']))
    # Encoder Hidden layer with sigmoid activation
    with tf.name_scope('layer2_encoder'):
        with tf.name_scope('weights'):
            variable_summaries(weights['encoder_h2'])
        with tf.name_scope('biases'):
            variable_summaries(biases['encoder_h2'])
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_h2']))

    return layer_2

def decoder(x):
    # decoder Hidden layer with sigmoid activation
    with tf.name_scope('layer1_decoder'):
        with tf.name_scope('weights'):
            variable_summaries(weights['decoder_h1'])
        with tf.name_scope('biases'):
            variable_summaries(biases['decoder_h1'])
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_h1']))
    # Encoder Hidden layer with sigmoid activation
    with tf.name_scope('layer2_decoder'):
        with tf.name_scope('weights'):
            variable_summaries(weights['decoder_h2'])
        with tf.name_scope('biases'):
            variable_summaries(biases['decoder_h2'])
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_h2']))
    return layer_2

# Set model
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

# prediction
y_pred=decoder_op
# True value
y_true=X

# define cost and the optimizer function
with tf.name_scope('cost'):
    cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
tf.summary.scalar('cost',cost)

optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()

with tf.Session() as sess:

    merged=tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/',sess.graph)

    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    # start training
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            summary,_,c=sess.run([merged,optimizer,cost],feed_dict={X:batch_xs})
            writer.add_summary(summary,i)
        if epoch % display_step==0:
            print("Epoch",'%04d' % (epoch+1),"cost:{}" .format(c))
    print("Finish")

    encoder_decoder=sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_to_show]})
    fig,a=plt.subplots(2,10,figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encoder_decoder[i],(28,28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()