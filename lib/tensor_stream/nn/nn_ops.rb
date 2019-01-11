module TensorStream
  # High level machine learning functions
  class NN
    extend TensorStream::OpHelper

    class << self
      def softmax(logits, axis: nil, name: nil)
        _op(:softmax, logits, axis: axis, name: name)
      end

      def relu(features, name: nil)
        TensorStream.max(features, 0, name: "relu_#{name}")
      end

      def relu6(features, name: nil)
        TensorStream.name_scope(name, "Relu6", values: [features]) do
          features = TensorStream.convert_to_tensor(features, name: "features")
          _op(:relu6, features, name: name)
        end
      end

      ##
      # Computes dropout.
      #
      # With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0. The scaling is so that the expected sum is unchanged.
      def dropout(x, keep_prob, noise_shape: nil, seed: nil, name: nil)
        TensorStream.name_scope(name, "dropout", values: [x]) do
          x = TensorStream.convert_to_tensor(x, name: "x")
          raise TensorStream::ValueError, "x has to be a floating point tensor since it's going to be scaled. Got a #{x.data_type} tensor instead." unless fp_type?(x.data_type)
          raise TensorStream::ValueError, "keep_prob must be a scalar tensor or a float in the range (0, 1], got #{keep_prob}" if keep_prob.is_a?(Float) && !(keep_prob > 0 && keep_prob <= 1)

          return x if keep_prob.is_a?(Float) && keep_prob.to_f == 1.0

          keep_prob = TensorStream.convert_to_tensor(keep_prob, dtype: x.dtype, name: "keep_prob")
          return x if keep_prob.value == 1.0

          noise_shape = if noise_shape.nil?
            TensorStream.shape(x)
          else
            noise_shape
          end

          random_tensor = keep_prob
          random_tensor += TensorStream.random_uniform(noise_shape, seed: seed, dtype: x.dtype)

          binary_tensor = TensorStream.floor(random_tensor)
          TensorStream.div(x, keep_prob) * binary_tensor
        end
      end

      def sigmoid(input, name: nil)
        TensorStream.sigmoid(input, name: name)
      end

      def softmax_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
        softmax_cross_entropy_with_logits_v2(labels: labels, logits: logits, name: name)
      end

      def softmax_cross_entropy_with_logits_v2(labels: nil, logits: nil, name: nil)
        TensorStream.name_scope(name, default: "softmax_cross_entropy_with_logits", values: [logits, labels]) do
          ts = TensorStream
          logits = ts.convert_to_tensor(logits, name: "logits")
          labels = ts.convert_to_tensor(labels, name: "labels")
          labels = ts.cast(labels, logits.dtype)

          output = _op(:softmax_cross_entropy_with_logits_v2, logits, labels)
          output[0]
        end
      end

      def sparse_softmax_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
        TensorStream.name_scope(name, default: "SparseSoftmaxCrossEntropyWithLogits", values: [logits, labels]) do
          tf = TensorStream
          labels = tf.convert_to_tensor(labels)
          logits = tf.convert_to_tensor(logits)
          precise_logits = logits.data_type == :float16 ? tf.cast(logits, :float32) : logits

          labels_static_shape = labels.shape
          labels_shape = tf.shape(labels)
          static_shapes_fully_defined = labels_static_shape.known? && logits.shape.known?

          raise TensorStream::ValueError, "Logits cannot be scalars - received shape #{logits.shape.shape}." if logits.shape.known? && logits.shape.scalar?
          if logits.shape.known? && (labels_static_shape.known? && labels_static_shape.ndims != logits.shape.ndims - 1)
            raise TensorStream::ValueError, "Rank mismatch: Rank of labels (received #{labels_static_shape.ndims}) " \
                                            "should equal rank of logits minus 1 (received #{logits.shape.ndims})."
          end
          if logits.shape.ndims == 2
            cost = _op(:sparse_softmax_cross_entropy_with_logits,
              precise_logits, labels, name: name)
            if logits.data_type == :float16
              return tf.cast(cost[0], :float16)
            else
              return cost[0]
            end
          end

          shape_checks = []

          shape_checks << tf.assert_equal(tf.rank(labels), tf.rank(logits) - 1) unless static_shapes_fully_defined

          tf.control_dependencies(shape_checks) do
            num_classes = tf.shape(logits)[tf.rank(logits) - 1]
            precise_logits = tf.reshape(precise_logits, [-1, num_classes])
            labels = tf.reshape(labels, [-1])
            cost = _op(:sparse_softmax_cross_entropy_with_logits, precise_logits, labels, name: name)
            cost = tf.reshape(cost[0], labels_shape)

            if logits.data_type == :float16
              tf.cast(cost, :float16)
            else
              cost
            end
          end
        end
      end

      # Computes log softmax activations.
      def log_softmax(logits, axis: -1, name: nil)
        _op(:log_softmax, logits, axis: axis, name: name)
      end

      def sigmoid_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
        TensorStream.name_scope(name, default: "logistic_loss", values: [logits, labels]) do |_name|
          tf = TensorStream
          logits = tf.convert_to_tensor(logits, name: "logits")
          labels = tf.convert_to_tensor(labels, name: "labels")
          zeros = tf.zeros_like(logits, dtype: logits.dtype)
          cond = (logits >= zeros)
          relu_logits = tf.where(cond, logits, zeros)
          neg_abs_logits = tf.where(cond, -logits, logits)

          tf.add(relu_logits - logits * labels,
            tf.log1p(tf.exp(neg_abs_logits)), name: name)
        end
      end

      def conv2d(input, filter, strides, padding, name: nil)
        _op(:conv2d, input, filter, strides: strides, padding: padding, name: name)
      end
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end
