module TensorStream
  # High level machine learning functions
  class NN
    extend TensorStream::OpHelper

    def self.softmax(logits, axis: nil, name: nil)
      _op(:softmax, logits, nil, axis: axis, name: name)
    end

    def self.relu(features, name: nil)
      TensorStream.max(features, 0, name: "relu_#{name}")
    end

    def self.relu6(features, name: nil)
      TensorStream.name_scope(name, "Relu6", values: [features]) do
        features = TensorStream.convert_to_tensor(features, name: "features")
        _op(:relu6, features, name: name)
      end
    end

    def self.sigmoid(input, name: nil)
      TensorStream.sigmoid(input, name: name)
    end

    def self.softmax_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
      softmax_cross_entropy_with_logits_v2(labels: labels, logits: logits, name: name)
    end

    def self.softmax_cross_entropy_with_logits_v2(labels: nil, logits: nil, name: nil)
      TensorStream.name_scope(name, default: 'softmax_cross_entropy_with_logits', values: [logits, labels]) do
        ts = TensorStream
        logits = ts.convert_to_tensor(logits, name: 'logits')
        labels = ts.convert_to_tensor(labels, name: 'labels')
        labels = ts.cast(labels, logits.dtype)

        output = _op(:softmax_cross_entropy_with_logits_v2, logits, labels)
        output[0]
      end
    end

    def self.sparse_softmax_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
      TensorStream.name_scope(name, default: "SparseSoftmaxCrossEntropyWithLogits", values: [logits, labels]) do
        tf = TensorStream
        labels = tf.convert_to_tensor(labels)
        logits = tf.convert_to_tensor(logits)
        precise_logits = logits.data_type == :float16 ? tf.cast(logits, :float32) : logits

        labels_static_shape = labels.shape
        labels_shape = tf.shape(labels)
        static_shapes_fully_defined = labels_static_shape.known? && logits.shape.known?

        raise TensorStream::ValueError, "Logits cannot be scalars - received shape #{logits.shape.shape}." if logits.shape.known? && logits.shape.scalar?
        raise TensorStream::ValueError, "Rank mismatch: Rank of labels (received #{labels_static_shape.ndims}) " +
          "should equal rank of logits minus 1 (received #{logits.shape.ndims})." if logits.shape.known? && (labels_static_shape.known? && labels_static_shape.ndims != logits.shape.ndims - 1)
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
    def self.log_softmax(logits, axis: -1, name: nil)
      _op(:log_softmax, logits, axis: axis, name: name)
    end

    def self.sigmoid_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
      TensorStream.name_scope(name, default: 'logistic_loss', values: [logits, labels]) do |_name|
        tf = TensorStream
        logits = tf.convert_to_tensor(logits, name: 'logits')
        labels = tf.convert_to_tensor(labels, name: 'labels')
        zeros = tf.zeros_like(logits, dtype: logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)

        tf.add(relu_logits - logits * labels,
               tf.log1p(tf.exp(neg_abs_logits)), name: name)
      end
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end
