require 'tensor_stream/nn/embedding_lookup'
require 'tensor_stream/utils/nn_utils'
module TensorStream
  # High level machine learning functions
  class NN
    extend TensorStream::OpHelper
    extend TensorStream::EmbeddingLookup
    extend TensorStream::Maths::MathFunctions
    extend TensorStream::TensorUtils
    extend TensorStream::NNUtils

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

      ##
      # Adds bias to value.
      #
      # This is a narrow version of tf add where the bias is restructed to 1-D only
      def bias_add(value, bias, data_format: nil, name: nil)
        value = TensorStream.convert_to_tensor(value, name: "input")
        bias = TensorStream.convert_to_tensor(bias, dtype: value.dtype, name: "bias")

        raise TensorStreamError, "value must be at least rank 2" if value.shape.known? && value.shape.ndims < 2

        _op(:bias_add, value, bias, data_format: data_format, name: name)
      end

      ##
      # Creates a recurrent neural network specified by RNNCell `cell`.
      # Performs fully dynamic unrolling of `inputs`.
      def dynamic_rnn(cell, inputs, sequence_length: nil, initial_state: nil,
        dtype: nil, parallel_iterations: nil, swap_memory: false,
        time_major: false, scope: nil)
        TensorStream.variable_scope(scope || "rnn") do |varscope|
          flat_input = _flatten(inputs)
          unless time_major
            flat_input = flat_input.map { |input| _transpose_batch_time(TensorStream.convert_to_tensor(input)) }.freeze
          end

          parallel_iterations = parallel_iterations || 32

          unless sequence_length.nil?
            sequence_length = TensorStream.cast(sequence_length, :int32)
            raise TensorStream::ValueError, "sequence_length must be a vector of length batch_size, " +
              "but saw shape: #{sequence_length.get_shape()}" unless [nil, 1].include?(sequence_length.shape.rank)
            sequence_length = TensorStream.identity(  # Just to find it in the graph.
                sequence_length, name="sequence_length")
          end

          batch_size = _best_effort_input_batch_size(flat_input)

          state = if !initial_state.nil?
                    initial_state
                  else
                    raise TensorStream::ValueError, "If there is no initial_state, you must give a dtype." unless dtype

                    if cell.respond_to?(:get_initial_state)
                      cell.get_initial_state(inputs: nil, batch_size: batch_size, dtype: dtype)
                    else
                      cell.zero_state(batch_size, dtype)
                    end
                  end
          inputs = pack_sequence_as(inputs, flat_input)

          outputs, final_state = _dynamic_rnn_loop(cell, inputs, state, parallel_iterations,
                                                                        swap_memory,
                                                                        sequence_length: sequence_length,
                                                                        dtype: dtype)

          outputs = map_structure(->(x) { _transpose_batch_time(x) }, outputs) unless time_major

          [outputs, final_state]
        end
      end

      ##
      # Infer the dtype of an RNN state.
      def _infer_state_dtype(explicit_dtype, state)
        return explicit_dtype if explicit_dtype

        if state.is_a?(Array)
          inferred_dtypes = _flatten(state).map { |element| element.dtype }
          raise TensorStream::ValueError, "Unable to infer dtype from empty state." if inferred_dtypes.empty?
          all_same = !inferred_dtypes.detect { |x| x != inferred_dtypes[0] }
          raise TensorStream::ValueError, "State has tensors of different inferred_dtypes. Unable to infer a single representative dtype." unless all_same
          return inferred_dtypes[0]
        end

        state.dtype
      end

      def _maybe_tensor_shape_from_tensor(shape)
        return TensorShape.as_shape(constant_value(shape)) if shape.is_a?(Tensor)

        shape
      end

      def _dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations, swap_memory, sequence_length: nil, dtype: nil)
        state = initial_state

        raise "parallel_iterations must be int" unless parallel_iterations.is_a?(Integer)

        state_size = cell.state_size
        flat_input = _flatten(inputs)
        flat_output_size = _flatten(cell.output_size)

        input_shape = TensorStream.shape(flat_input[0])
        time_steps = input_shape[0]
        batch_size = _best_effort_input_batch_size(flat_input)

        inputs_got_shape = flat_input.map { |inp| inp.shape.with_rank_at_least(3) }.freeze
        const_time_steps, const_batch_size = inputs_got_shape[0].shape[0..2]
        inputs_got_shape.each do |shape|
          raise TensorStream::ValueError, "Input size (depth of inputs) must be accessible via shape inference, but saw value None." unless shape[2..shape.shape.size].fully_defined?

          got_time_steps = shape[0].value
          got_batch_size = shape[1].value

          raise TensorStream::ValueError, "Time steps is not the same for all the elements in the input in a batch." if const_time_steps != got_time_steps
          raise TensorStream::ValueError, "Batch_size is not the same for all the elements in the input." if const_batch_size != got_batch_size

          zero_array_creator = ->(size) {
            size = _concat(batch_size, size)
            TensorStream.zeros(TensorStream.stack(size), dtype: _infer_state_dtype(dtype, state))
          }

          flat_zero_output = flat_output_size.map { |output| zero_array_creator.call(output) }.freeze

          zero_output = pack_sequence_as(cell.output_size, flat_zero_output)

          min_sequence_length,  max_sequence_length = if sequence_length
                                                        [TensorStream.reduce_min(sequence_length), TensorStream.reduce_max(sequence_length)]
                                                      else
                                                        [nil, time_steps]
                                                      end
          time = TensorStream.constant(0, dtype: :int32, name: 'time')

          base_name = TensorStream.name_scope("dynamic_rnn") { |scope| scope }

          ta_creator = ->(name, element_shape, dtype) {
            TensorStream::TensorArray.new(dtype, time_steps: time_steps, element_shape: element_shape, tensor_array_name: base_name + name)
          }

          output_ta = flat_output_size.each_with_index.map do |out_size, i|
            ta_creator.call("output_#{i}", TensorShape.new([const_batch_size]).concatenate(_maybe_tensor_shape_from_tensor(out_size)), _infer_state_dtype(dtype, state))
          end.freeze

          input_ta = flat_input.each_with_index.map do |flat_input_i, i|
            ta_creator.call("input_#{i}", flat_input_i.shape[1..flat_input_i.shape.shape.size], flat_input_i.dtype)
          end.freeze

          input_ta = input_ta.zip(flat_inpute).map { |ta, inp| ta.unstack(inp) }.freeze

          time_step_fn = -> (time, output_ta_t, state) {
            input_t = input_ta.map { |ta| ta.read(time) }.freeze
          }

          input_t = pack_sequence_as(inputs, input_t)
        end
      end
    end
  end

  # tensorflow compatibility
  def self.nn
    TensorStream::NN
  end
end
