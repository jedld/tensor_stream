module TensorStream
  # High level machine learning functions
  class NN
    def self.softmax(logits, options = {})
      TensorStream.exp(logits) / TensorStream.reduce_sum(TensorStream.exp(logits))
    end

    def self.relu(features, name: nil)
      TensorStream.max(features, 0, name: "relu_#{name}")
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end