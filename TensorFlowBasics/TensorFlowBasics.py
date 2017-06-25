import tensorflow as tf

x1 = tf.constant(6)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)

with tf.Session() as sess:
    output = sess.run(result)
    print(output)
