require 'tensor_stream/train/slot_creator'
require 'tensor_stream/train/optimizer'
require 'tensor_stream/train/gradient_descent_optimizer'
require 'tensor_stream/train/momentum_optimizer'
require 'tensor_stream/train/adam_optimizer'
require 'tensor_stream/train/adadelta_optimizer'
require 'tensor_stream/train/saver'

module TensorStream
  module Trainer
    extend TensorStream::Train::Utils

    def self.write_graph(graph, path, filename, as_text: true, serializer: TensorStream::Pbtext)
      raise "only supports as_text=true for now" unless as_text
      new_filename = File.join(path, filename)
      File.write(new_filename, serializer.new.get_string(graph))
    end
  end
end
