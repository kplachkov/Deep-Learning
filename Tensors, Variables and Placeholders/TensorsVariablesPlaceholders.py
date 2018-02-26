"""A tensor is multidimensional array"""

import tensorflow as tf

scalar = tf.constant([3])
vector = tf.constant([3, 4, 5, 6])
matrix = tf.constant([[3, 5], [9, 10]])
tensor = tf.constant([[[3, 5], [9, 10]], [[7, 5], [1, 10]]])

matrix2 = tf.constant([[7, 1], [3, 10]])

"""Operations with matrices."""
result_adding_matrices = matrix + matrix2

with tf.Session() as session:
    result = session.run(scalar)
    print("Scalar:\n", result)
    result = session.run(vector)
    print("Vector:\n", result)
    result = session.run(matrix)
    print("Matrix:\n", result)
    result = session.run(tensor)
    print("Tensor:\n", result)
    result = session.run(result_adding_matrices)
    print("Adding two matrices:\n", result)

"""Counter."""
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.variables_initializer([state])
print("Counter:")
with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(5):
        session.run(update)
        print(session.run(state))

"""Placeholders. Feed data to TensorFlow from outside a model. Placeholder is like
variable that will not actually receive its data until a later point"""

a = tf.placeholder(tf.float32)
b = a * 2

with tf.Session() as session:
    result = session.run(b, feed_dict={a: 6.5})
    print("Placeholder:", result)

dictionary = {a: [[[3, 5], [9, 10]], [[7, 5], [1, 10]]]}
with tf.Session() as session:
    result = session.run(b, feed_dict=dictionary)
    print("Placeholder\n", result)

