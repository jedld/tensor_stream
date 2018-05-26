import tensorflow as tf

x = tf.constant([[1.0, 0.5, 4.0]])

w = tf.constant([[0.4, 0.2],[0.1, 0.45],[0.2, 4.0]])

w2 = tf.constant([[0.3, 0.2],[0.15, 0.45]])
w3 = tf.constant([[0.1, 0.1, 1.0, 1.1, 0.4],[0.05, 0.2, 1.0, 1.2, 0.5],])

b= tf.constant([4.0, 5.0])
b2= tf.constant([4.1, 5.1])
b3 = tf.constant([2.0, 3.1, 1.0, 0.2, 0.2])

matmul_layer_1 = tf.matmul(x, w)
a = tf.sin(matmul_layer_1 + b)
matmul_layer_2 = tf.matmul(a, w2)
matmul_layer_2_add = matmul_layer_2 + b2
a2 = tf.sin(matmul_layer_2_add)



g_matmul_layer_1 =  tf.gradients(matmul_layer_1, [x, w])
g_sin_a = tf.gradients(a, [b])
g_matmul_layer_2 =  tf.gradients(matmul_layer_2, [b])
g_matmul_layer_2_add = tf.gradients(matmul_layer_2_add, [b])

sess = tf.Session()
s2 = sess.run(g_matmul_layer_2_add)
g_a2 = tf.gradients(a2, [b], name="final")

print("layer_1 %s", sess.run(g_matmul_layer_1))
print("layer_2 %s", sess.run(g_matmul_layer_2))
print("matmul_layer_2_add %s", s2)
print("g_sin_a %s", sess.run(g_sin_a))
print("-- %s", sess.run(tf.cos(matmul_layer_2_add) * g_matmul_layer_2_add))
print("%s", sess.run(g_a2))

writer = tf.summary.FileWriter("/home/jedld/graphs/", sess.graph)
sess.run(g_a2)
writer.close()

