import tensorflow as tf

batch_x = [
          [0.686274, 0.10196, 0.6509, 1.0, 0.9686, 0.49803, 0.0, 0.0, 0.0, 0.0],
          [0.543244, 0.10123, 0.4509, 0.0, 0.6986, 0.39803, 1.0, 0.0, 0.0, 0.0]]

batch_y = [
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
]

num_input = 10
num_classes = 10
n_hidden_1 = 4 # 1st layer number of neurons
n_hidden_2 = 4 # 2nd layer number of neurons

X = batch_x #tf.placeholder(tf.float32, shape=[None, num_input])
Y = batch_y # tf.placeholder(tf.float32, shape=[None, num_classes])

h1_init = tf.constant([[0.5937, 0.2343, 1.4332, 0.4395],
          [-1.0227, -0.6915, 1.2367, 0.3452],
          [-0.5675, 1.0374, 1.0429, 0.8839],
          [-0.1066, -0.0469, -1.6317, -1.4836],
          [0.7835, -3.0105, 1.713, -0.4536],
          [-0.3076, 1.3662, -0.6537, 0.0905],
          [-0.2459, 0.2243, -2.7048, 0.848],
          [0.3589, 0.3542, -0.0959, -1.327],
          [-0.4685, 0.0844, 0.2794, 2.1275],
          [-1.0733, 0.6189, 0.845, 0.033]])

h2_init = tf.constant([[0.5012, 1.302, -1.6217, 0.669], [0.1494, -0.7837, -0.2978, 1.7745], [1.9727, -0.5312, -0.7391, 0.9187], [-0.6412, -1.4434, -0.8801, 0.9343]])
h3_init = tf.constant([[0.5012, 1.302, -1.6217, 0.669, 0.1494, -0.7837, -0.2978, 1.7745, 1.9727, -0.5312],
  [-0.7391, 0.9187, -0.6412, -1.4434, -0.8801, 0.9343, -0.1665, -0.0032, 0.2959, -2.0488],
  [-0.9135, 1.0376, 0.8537, 0.4376, 1.3255, -0.5921, -1.4081, 1.0614, -0.5283, 1.1832],
  [0.7285, -0.7844, 0.1793, -0.5275, -0.4426, -1.4976, 0.4433, 2.2317, -2.0479, 0.7791]])


b1_init = tf.constant([0.1494, -0.7837, -0.2978, 1.7745])

b2_init = tf.constant([1.9727, -0.5312, -0.7391, 0.9187])
out_init = tf.constant([-0.6412, -1.4434, -0.8801, 0.9343, -0.1665, -0.0032, 0.2959, -2.0488, -0.9135, 1.0376])

h1 = tf.Variable(h1_init, dtype=tf.float32, name='h1')
h2 = tf.Variable(h2_init, dtype=tf.float32, name='h2')
h3 = tf.Variable(h3_init, dtype=tf.float32, name='out')

b1 = tf.Variable(b1_init, dtype=tf.float32, name='b1')
b2 = tf.Variable(b2_init, dtype=tf.float32, name='b2')
out = tf.Variable(out_init, dtype=tf.float32, name='out2')

layer_1 = tf.add(tf.matmul(X, h1), b1)
# Hidden fully connected layer with 256 neurons
layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
# Output fully connected layer with a neuron for each class

sess = tf.Session()

logits = tf.matmul(layer_2, h3) + out
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

sess.run(init)
# print(sess.run(layer_1))
tf.gradients(loss_op, [logits])
print("------------")

print(sess.run(h1))

# sess.run(train_op, feed_dict={ X: batch_x, Y: batch_y })
sess.run(train_op)
print(sess.run(h1))
print(sess.run(h2))
print(sess.run(h3))

print(sess.run(b1))
print(sess.run(b2))
print(sess.run(out))

# sess.run(train_op, feed_dict={ X: batch_x, Y: batch_y })
sess.run(train_op)
print(sess.run(h1))