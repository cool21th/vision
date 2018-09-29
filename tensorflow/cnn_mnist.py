import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)
keep_rate = tf.placeholder(tf.float32)

w_conv1 = tf.Variable(tf.truncated_normal([3,3,1,32], stddev=0.1))
w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
w_full3 = tf.Variable(tf.truncated_normal([7*7*64, 256], stddev=0.1))
w_full4 = tf.Variable(tf.truncated_normal([256,10], stddev=0.1))

b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
b_full3 = tf.Variable(tf.constant(0.1, shape=[256]))
b_full4 = tf.Variable(tf.constant(0.1, shape=[10]))

x_image = tf.reshape(x, [-1, 28, 28, 1])

c_conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
print(c_conv1.shape)
h_conv1 = tf.nn.relu(c_conv1 + b_conv1)
print(h_conv1.shape)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(h_pool1.shape)

c_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
print(c_conv2.shape)

h_conv2 = tf.nn.relu(c_conv2 + b_conv2)
print(h_conv2.shape)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(h_pool2.shape)

h_flats = tf.reshape(h_pool2, [-1, 7*7 * 64])
print(h_flats.shape)

h_full3 = tf.nn.relu(tf.matmul(h_flats, w_full3) + b_full3)
h_drop3 = tf.nn.dropout(h_full3, keep_rate)
print(h_full3.shape)

h_full4 = tf.matmul(h_drop3, w_full4) + b_full4

print(h_full4.shape)
hx = h_full4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_full4,labels=y))

optimizer  = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 3
batch_size = 100
n_iters = len(mnist.train.images) //batch_size

for i in range(epochs):
    loss = 0
    for j in range(n_iters):
        xx, yy = mnist.train.next_batch(batch_size)
        sess.run(train , {x: xx, y: yy, keep_rate:0.5})
        loss += sess.run(cost, {x: xx, y:yy, keep_rate:0.5})

    print(i, loss / n_iters)

print('-' * 50)

pred = sess.run(hx, {x: mnist.test.images, keep_rate:1.0})
pred_arg = np.argmax(pred, axis=1)

test_arg = np.argmax(mnist.test.labels, axis=1)

print('acc : ', np.mean(pred_arg == test_arg))
sess.close()
