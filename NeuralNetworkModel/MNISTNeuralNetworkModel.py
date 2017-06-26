import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST/data", one_hot=True)

# 10 classes, 0-9
"""
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
..................................
"""

n_nodes_hidden_layer1 = 500
n_nodes_hidden_layer2 = 500
n_nodes_hidden_layer3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")


def neural_network_model(data):
    hidden_layer1 = {"weights": tf.Variable(tf.random_normal([784, n_nodes_hidden_layer1])),
                     "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer1]))}

    hidden_layer2 = {"weights": tf.Variable(tf.random_normal([n_nodes_hidden_layer1, n_nodes_hidden_layer2])),
                     "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer2]))}

    hidden_layer3 = {"weights": tf.Variable(tf.random_normal([n_nodes_hidden_layer2, n_nodes_hidden_layer3])),
                     "biases": tf.Variable(tf.random_normal([n_nodes_hidden_layer3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hidden_layer3, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    layer1 = tf.add(tf.matmul(data, hidden_layer1["weights"]), hidden_layer1["biases"])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer2["weights"]), hidden_layer2["biases"])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, hidden_layer3["weights"]), hidden_layer3["biases"])
    layer3 = tf.nn.relu(layer3)

    output = tf.matmul(layer3, output_layer["weights"]) + output_layer["biases"]

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10  # cycles feed forward + backprop

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch, "completed out of", hm_epochs, "loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
