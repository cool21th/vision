import tflearn
from tensorflow.examples.tutorials.mnist import input_data

net = tflearn.input_data([None, 28,28,1])

net = tflearn.conv_2d(net, nb_filter=6, filter_size=5,
                      strides=1, padding='SAME', activation='sigmoid')
net = tflearn.avg_pool_2d(net, kernel_size=2, strides=2, padding='SAME')

net = tflearn.conv_2d(net, nb_filter=16, filter_size=5,
                      strides=1, padding='VALID', activation='sigmoid')
net = tflearn.avg_pool_2d(net, kernel_size=2, strides=2, padding='SAME')

net = tflearn.fully_connected(net, n_units=120, activation='sigmoid')
net = tflearn.fully_connected(net, n_units=84, activation='sigmoid')
net = tflearn.fully_connected(net, n_units=10, activation='softmax')

net = tflearn.regression(net, learning_rate=0.001, optimizer='rmsprop', loss='categorical_crossentropy')

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = mnist.train.images.reshape(-1, 28, 28, 1)
y = mnist.train.labels

model = tflearn.DNN(net)
model.fit(x, y, n_epoch=10, batch_size=128, shuffle=True, validation_set=0.2)

x = mnist.test.images.reshape(-1, 28, 28, 1)
y = mnist.test.labels

acc = model.evaluate(x, y)

print('acc: ', acc)
