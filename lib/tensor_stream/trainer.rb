require 'tensor_stream/train/slot_creator'
require 'tensor_stream/train/optimizer'
require 'tensor_stream/train/gradient_descent_optimizer'
require 'tensor_stream/train/momentum_optimizer'
require 'tensor_stream/train/adam_optimizer'
require 'tensor_stream/train/adadelta_optimizer'
require 'tensor_stream/train/adagrad_optimizer'
require 'tensor_stream/train/rmsprop_optimizer'
require 'tensor_stream/train/saver'
require 'tensor_stream/train/learning_rate_decay'

module TensorStream
  module Trainer
    extend TensorStream::Train::Utils
    extend TensorStream::Train::LearningRateDecay
    extend TensorStream::StringHelper

    def self.write_graph(graph, path, filename, as_text: true, serializer: :yaml)
      raise "only supports as_text=true for now" unless as_text

      serializer = constantize("TensorStream::#{camelize(serializer.to_s)}") if serializer.is_a?(Symbol)

      new_filename = File.join(path, filename)
      serializer.new.get_string(graph).tap do |str|
        File.write(new_filename, str)
      end
    end
  end
end
