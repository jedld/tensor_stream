require 'tensor_stream/version'
require 'deep_merge'
require 'matrix'
require 'concurrent'
require 'tensor_stream/helpers/op_helper'
require 'tensor_stream/initializer'
require 'tensor_stream/graph_keys'
require 'tensor_stream/types'
require 'tensor_stream/graph'
require 'tensor_stream/device'
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
require 'tensor_stream/monkey_patches/integer'
require 'tensor_stream/ops'

# module that exposes TensorStream top level functions
module TensorStream
  extend TensorStream::OpHelper
  extend TensorStream::Ops

  def self.float32
    Types.float32
  end

  def self.graph
    TensorStream::Graph.new
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

  def self.variable(value, name: nil, initializer: nil, graph: nil, dtype: nil, trainable: true)
    common_options = {
      initializer: initializer || Operation.new(:assign, nil, value),
      name: name,
      graph: graph,
      dtype: dtype,
      trainable: trainable
    }
    if value.is_a?(String)
      TensorStream::Variable.new(dtype || :string, 0, [], common_options)
    elsif value.is_a?(Integer)
      TensorStream::Variable.new(dtype || :int32, 0, [], common_options)
    elsif value.is_a?(Float)
      TensorStream::Variable.new(dtype || :float32, 0, [], common_options)
    else
      TensorStream::Variable.new(dtype || :float32, 0, nil, common_options)
    end
  end

  def self.variable_scope(scope = nil, reuse: nil, initializer: nil)
    Thread.current[:tensor_stream_variable_scope] ||= []
    Thread.current[:tensor_stream_variable_scope] << OpenStruct.new(name: scope, reuse: reuse, initializer: initializer)
    scope_name = __v_scope_name
    begin
      if block_given?
        TensorStream.get_default_graph.name_scope(scope) do
          yield(scope_name)
        end
      end
    ensure
      Thread.current[:tensor_stream_variable_scope].pop
    end
  end

  def self.get_variable_scope
    return nil unless Thread.current[:tensor_stream_variable_scope]
    __v_scope_name
  end

  def self.__v_scope_name
    Thread.current[:tensor_stream_variable_scope].map(&:name).compact.reject(&:empty?).join('/')
  end

  def self.session(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor)
    session = TensorStream::Session.new(evaluator, thread_pool_class: thread_pool_class)
    yield session if block_given?

    session
  end

  def self.program(&block)
    block.call(self)
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

      Kernel.loop do
        dtype, rank, value_ptr, d = dtype_eval(rank, value_ptr)
        dimensions << d
        break if dtype != :array
      end

      TensorStream::Tensor.new(dtype, rank, options[:shape] || dimensions, shared_options)
    end
  end

  def self.group(inputs)
    TensorStream::ControlFlow.new(:group, inputs)
  end

  def self.get_variable(name, dtype: nil, shape: nil, initializer: nil, trainable: true, collections: nil)
    TensorStream::Variable.new(dtype || :float32, nil, shape, collections: collections, name: name, initializer: initializer, trainable: trainable)
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

  def self.trainable_variables
    TensorStream.get_default_graph.get_collection(TensorStream::GraphKeys::TRAINABLE_VARIABLES)
  end

  def self.set_random_seed(seed)
    TensorStream.get_default_graph.random_seed = seed
  end

  def self.check_allowed_types(input, types)
    return input unless input.is_a?(Tensor)
    return input if input.data_type.nil?

    raise "Parameter data type #{input.data_type} passed not in #{types.join(',')}" unless types.map(&:to_sym).include?(input.data_type)
  end
end
