module TensorStream
  # High level machine learning functions
  class NN
    def self.softmax(logits, _options = {})
      TensorStream.exp(logits) / TensorStream.reduce_sum(TensorStream.exp(logits))
    end

    def self.relu(features, name: nil)
      TensorStream.max(features, 0, name: "relu_#{name}")
    end

    def self.sigmoid_cross_entropy_with_logits(labels: nil, logits: nil, name: nil)
      TensorStream.name_scope(name, default: 'logistic_loss', values: [logits, labels]) do |name|
        tf = TensorStream
        logits = tf.convert_to_tensor(logits, name: 'logits')
        labels = tf.convert_to_tensor(labels, name: 'labels')
        zeros = tf.zeros_like(logits, dtype: logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        return tf.add(
            relu_logits - logits * labels,
            tf.log1p(tf.exp(neg_abs_logits)),
            name: name)
      end
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end
