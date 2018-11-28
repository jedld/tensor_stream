import tensorflow as tf

test_inputs = [
  [0.5937, 0.2343, 1.4332, 0.4395],
  [-1.0227, -0.6915, 1.2367, 0.3452],
  [-0.5675, 1.0374, 1.0429, 0.8839],
  [-0.1066, -0.0469, -1.6317, -1.4836],
  [0.7835, -3.0105, 1.713, -0.4536],
  [-0.3076, 1.3662, -0.6537, 0.0905],
  [-0.2459, 0.2243, -2.7048, 0.848],
]

num_inputs = 4
num_neurons = 5
inputs = tf.placeholder("float", shape=(None, num_inputs))
biases = tf.constant([0.5012, 1.302, -1.6217, 0.669, 0.1494], name='b1')
biases2 = tf.constant([0.2012, 1.102, -1.5217, 0.469, 0.0494], name='b2')

weights = tf.constant([
  [-0.9135, 1.0376, 0.8537, 0.4376, 1.3255],
  [-0.5921, -1.4081, 1.0614, -0.5283, 1.1832],
  [0.7285, -0.7844, 0.1793, -0.5275, -0.4426],
  [-1.4976, 0.4433, 2.2317, -2.0479, 0.7791]], name='w')

weights_layer2 = tf.constant([
  [-1.0465, -0.8766, 1.6849, -0.6625, 0.7928],
  [2.0412, 1.3564, 0.7905, 0.6434, -2.5495],
  [2.4276, -0.6893, -1.5917, 0.0911, 0.9112],
  [-0.012, 0.0794, 1.3829, -1.018, -0.9328],
  [0.061, 0.9791, -2.1727, -0.9553, -1.434]], name='w2')


sess = tf.Session()

layer_1 =  tf.matmul(inputs, weights) + biases
neural_net = tf.matmul(layer_1, weights_layer2) + biases2

output = sess.run(neural_net, feed_dict={ inputs: test_inputs })

g0 = tf.gradients(layer_1, [weights, biases])
g = tf.gradients(neural_net, [weights, biases])
g2 = tf.gradients(neural_net, [weights_layer2, biases2])

weight_gradient0, biases_gradient0 = sess.run(g0, feed_dict = { inputs: test_inputs })
weight_gradient, biases_gradient = sess.run(g, feed_dict = { inputs: test_inputs })
weight_gradient2, biases_gradient2 = sess.run(g2, feed_dict = { inputs => test_inputs })
