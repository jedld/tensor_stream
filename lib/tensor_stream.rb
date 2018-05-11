require "tensor_stream/version"
require 'deep_merge'
require 'matrix'
require 'concurrent'
require 'tensor_stream/helpers/op_helper'
require 'tensor_stream/graph_keys'
require 'tensor_stream/types'
require 'tensor_stream/graph'
require 'tensor_stream/session'
require 'tensor_stream/tensor_shape'
require 'tensor_stream/tensor'
require 'tensor_stream/variable'
require 'tensor_stream/operation'
require 'tensor_stream/placeholder'
require 'tensor_stream/control_flow'
require 'tensor_stream/trainer'
require 'tensor_stream/nn/nn_ops'
require 'tensor_stream/evaluator/evaluator'
# require 'tensor_stream/libraries/layers'
require "tensor_stream/monkey_patches/integer"
require 'tensor_stream/ops'

module TensorStream
  extend TensorStream::OpHelper
  extend TensorStream::Ops

  def self.float32
    Types.float32
  end

  def self.get_default_graph
    TensorStream::Graph.get_default_graph
  end

  def self.reset_default_graph
    TensorStream::Graph.get_default_graph.reset
  end

  def self.enable_eager_execution
    TensorStream::Graph.get_default_graph.enable_eager_execution
  end

  def self.disable_eager_execution
    TensorStream::Graph.get_default_graph.disable_eager_execution
  end

  def self.executing_eagerly?
    TensorStream::Graph.get_default_graph.executing_eagerly?
  end

  def self.Variable(value, options = {})
    common_options= {
      initializer: Operation.new(:assign, nil, value),
      name: options[:name]
    }
    if value.is_a?(String)
      TensorStream::Variable.new(options[:dtype] || :string, 0, [], common_options)
    elsif value.is_a?(Integer)
      TensorStream::Variable.new(options[:dtype] || :int32, 0, [], common_options)
    elsif value.is_a?(Float)
      TensorStream::Variable.new(options[:dtype] || :float32, 0, [], common_options)
    else
      TensorStream::Variable.new(options[:dtype] || :float32, 0, nil, common_options)
    end
  end

  def self.Session(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor)
    session = TensorStream::Session.new(evaluator, thread_pool_class: thread_pool_class)
    if block_given?
      yield session
    end
    session
  end

  def self.program(&block)
    block.(self)
  end

  def self.layers
    TensorStream::Layers
  end

  def self.constant(value, options = {})
    shared_options = { const: true, value: value, name: options[:name] }
    if value.is_a?(Float)
      TensorStream::Tensor.new(options[:dtype] || :float32, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(Integer)
      TensorStream::Tensor.new(options[:dtype] || :int32, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(String)
      TensorStream::Tensor.new(options[:dtype] || :string, 0, options[:shape] || [], shared_options)
    elsif value.is_a?(Array)
      dtype = nil
      rank = 1
      dimensions = []
      value_ptr = value

      begin
        dtype, rank, value_ptr, d = dtype_eval(dtype, rank, value_ptr)
        dimensions << d
      end while dtype == :array

      TensorStream::Tensor.new(dtype, rank, options[:shape] || dimensions, shared_options)
    end
  end

  def self.group(inputs)
    TensorStream::ControlFlow.new(:group, inputs)
  end

  def self.get_variable(name, options = {})
    TensorStream::Variable.new(options[:dtype] || :float32, nil, options[:shape], name: name, initializer: options[:initializer])
  end

  def self.get_collection(name, options = {})
    Graph.get_default_graph.get_collection(name, options)
  end

  def self.placeholder(dtype, options = {})
    TensorStream::Placeholder.new(dtype, nil, options[:shape])
  end

  def self.global_variables_initializer
    TensorStream::Variable.global_variables_initializer
  end

  def self.train
    TensorStream::Trainer
  end

  private

  def self.check_allowed_types(t, types)
    return t unless t.is_a?(Tensor)
    return t if t.data_type.nil?

    fail "Parameter data type #{t.data_type} passed not in #{types.join(',')}" if !types.map(&:to_sym).include?(t.data_type)
  end
end
