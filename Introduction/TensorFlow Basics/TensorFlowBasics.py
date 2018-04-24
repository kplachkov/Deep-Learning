import tensorflow as tf

a = tf.constant(6)
b = tf.constant(6)

c = tf.multiply(a, b)

with tf.Session() as session:
    result = session.run(c)
    print(result)
