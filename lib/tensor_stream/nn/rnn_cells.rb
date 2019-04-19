##
# This meant a ruby port of tensorflow/python/ops/rnn_cell_impl.py credited to TensorFlow Authors.
# The original python source file this is based on falls under the Apache License, Version 2.0
#
# However it is not expected that translation is not perfect as there are changes needed in order to follow
# ruby conventions

require 'tensor_stream/layer/tensor_array'
module TensorStream
  module RNN
    class RNNCell < TensorStream::Layer
      include TensorStream::TensorUtils
      include TensorStream::NNUtils

      def initialize(options = {}, trainable: true, name: nil, dtype: nil)
        super(options, trainable: trainable, name: name, dtype: dtype)
        @is_tf_rnn_cell = true
      end

      def get_initial_state(inputs: nil, batch_size: nil, dtype: nil)
        unless inputs.nil?
          inputs = TensorStream.convert_to_tensor(inputs, name: "inputs")
          unless batch_size.nil?
            static_batch_size = if batch_size.is_a?(Tensor)
                                  constant_value(batch_size, partial: true)
                                else
                                  batch_size
                                end
            raise TensorStream::ValueError, "batch size from input tensor is different from the " +
              "input param. Input tensor batch: #{inputs.shape.dims[0].value}, batch_size: #{batch_size}"
          end

          raise TensorStream::ValueError,
            "dtype from input tensor is different from the " +
            "input param. Input tensor dtype: #{inputs.dtype}, dtype: #{dtype}" if !dtype.nil? && inputs.dtype != dtype
          batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
          dtype = inputs.dtype
        end

        raise TensorStream::ValueError,
          "batch_size and dtype cannot be nil while constructing initial " +
          "state: batch_size=#{batch_size}, dtype=#{dtype}" if batch_size.nil? || dtype.nil?
        zero_state(batch_size, dtype)
      end

      def zero_state(batch_size, dtype)
        state_size = self.state_size
        TensorStream.name_scope("#{@name}ZeroState", [batch_size]) do
          _zero_state_tensors(state_size, batch_size, dtype)
        end
      end

      def state_size
        raise "Abstract method"
      end

      protected

      ##
      # Concat that enables int, Tensor, or TensorShape values.
      # This function takes a size specification, which can be an integer, a
      # TensorShape, or a Tensor, and converts it into a concatenated Tensor
      # (if static = False) or a list of integers (if static = True).
      # Args:
      # prefix: The prefix; usually the batch size (and/or time step size).
      #  (TensorShape, int, or Tensor.)
      # suffix: TensorShape, int, or Tensor.
      # static: If `True`, return a python list with possibly unknown dimensions.
      #  Otherwise return a `Tensor`.
      # Returns:
      #   shape: the concatenation of prefix and suffix.
      # Raises:
      # ValueError: if `suffix` is not a scalar or vector (or TensorShape).
      # ValueError: if prefix or suffix was `None` and asked for dynamic
      #      Tensors out.
      def concat(prefix, suffix, static: false)
      #   if prefix.is_a?(Tensor)
      #     p = prefix
      #     p_static = constant_value(prefix)
      #     if p.shape.ndims == 0:
      #   p = array_ops.expand_dims(p, 0)
      # elif p.shape.ndims != 1:
      #   raise ValueError("prefix tensor must be either a scalar or vector, "
      #                    "but saw tensor: %s" % p)
      #   end
      end
    end

    class MultiRNNCell < RNNCell
      ##
      # Create a RNN cell composed sequentially of a number of RNNCells.
      def initialize(cells, state_is_tuple: true)
        super()

        raise TensorStream::ValueError, "Must specify at least one cell for MultiRNNCell" unless cells
        raise TensorStream::TypeError, "cells must be a list, but saw: #{cells.class.name}" unless cells.is_a?(Array)

        @cells = cells
        @cells.each_with_index do |cell, cell_number|
          if cell.is_a?(TensorStream::Trackable)
            track_trackable(cell, name: "cell-#{cell_number}")
          end
        end

        @state_is_tuple = state_is_tuple
      end

      def state_size
        if @state_is_tuple
          @cells.map { |cell| cell.state_size }.freeze
        else
          @cells.map { |cell| cell.state_size }.reduce(:+)
        end
      end

      def output_size
        @cells.last.output_size
      end

      def zero_state(batch_size, dtype)
        TensorStream.name_scope(self.class.name + "ZeroState", values: [batch_size]) do
          if @state_is_tuple
            @cells.map { |cell| cell.zero_state(batch_size, dtype) }.freeze
          else
            super.zero_state(batch_size, dtype)
          end
        end
      end
    end

    class LayerRNNCell < RNNCell
    end

    class GRUCell < LayerRNNCell
      def initialize(num_units, options = {}, activation: nil, reuse: nil, kernel_initializer: nil,
        bias_initializer: nil, name: nil, dtype: nil)
        @num_units = num_units
        @activation = activation ? activation : :tanh
        @kernel_initializer = kernel_initializer
        @bias_initializer = bias_initializer
      end

      def state_size
        @num_units
      end

      def output_size
        @num_units
      end
    end
  end
end