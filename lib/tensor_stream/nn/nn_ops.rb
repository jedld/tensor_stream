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

    def self.sigmoid(input, name: nil)
      TensorStream.sigmoid(input, name: name)
    end

    def self.softmax_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
      softmax_cross_entropy_with_logits_v2(labels: labels, logits: logits, name: name)
    end

    def self.softmax_cross_entropy_with_logits_v2(labels: nil, logits: nil, name: nil)
      TensorStream.name_scope(name, default: 'softmax_cross_entropy_with_logits', values: [logits, labels]) do
        tf = TensorStream
        logits = tf.convert_to_tensor(logits, name: 'logits')
        labels = tf.convert_to_tensor(labels, name: 'labels')
        labels = tf.cast(labels, logits.dtype)
        output = _op(:softmax_cross_entropy_with_logits_v2, logits, labels)
        output[0]
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
