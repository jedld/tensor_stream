module TensorStream
  # Class that defines all available ops supported by TensorStream
  module Ops
    FLOATING_POINT_TYPES = %w[float32 float64].map(&:to_sym)
    NUMERIC_TYPES = %w[int32 int64 float32 float64].map(&:to_sym)

    def argmax(input, axis = nil, name: nil, dimension: nil, output_type: :int32)
      _op(:argmax, input, nil, axis: axis, name: name, dimension: dimension, data_type: output_type)
    end

    def gradients(input, wrt_xs, grad_ys: nil,
                  name: 'gradients',
                  colocate_gradients_with_ops: false,
                  gate_gradients: false,
                  aggregation_method: nil,
                  stop_gradients: nil)

      gs = wrt_xs.collect do |x|
        raise "#{x} passed is not a tensor object" unless x.is_a?(Tensor)

        stops = stop_gradients ? stop_gradients.map(&:name).join('_') : ''
        gradient_program_name = "grad_#{input.name}_#{x.name}_#{stops}".to_sym

        tensor_program = if input.graph.node_added?(gradient_program_name)
                           input.graph.get_node(gradient_program_name)
                         else
                           derivative_ops = TensorStream::MathGradients.derivative(input, x, graph: input.graph,
                                                                                             stop_gradients: stop_gradients)
                           unit_matrix = _op(:ones_like, x)
                           input.graph.add_node!(gradient_program_name, unit_matrix * derivative_ops)
                         end
        tensor_program
      end
      TensorStream.group(gs)
    end

    def random_uniform(shape, dtype: :float32, minval: 0, maxval: 1, seed: nil, name: nil)
      options = { shape: shape, dtype: dtype, minval: minval, maxval: maxval, seed: seed, name: name }
      _op(:random_uniform, nil, nil, options)
    end

    def random_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
      options = { shape: shape, dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name }
      _op(:random_normal, nil, nil, options)
    end

    def stop_gradient(tensor, options = {})
      _op(:stop_gradient, tensor, nil, options)
    end

    def eye(num_rows, num_columns: nil, dtype: :float32, name: nil)
      _op(:eye, num_rows, num_columns || num_rows, data_type: dtype, name: name)
    end

    def shape(input, name: nil, out_type: :int32)
      _op(:shape, input, nil, name: name, out_type: out_type)
    end

    def rank(input, name: nil)
      _op(:rank, input, name: name)
    end

    def zeros_initializer(options = {})
      _op(:zeros, nil, nil, options)
    end

    def glorot_uniform_initializer(seed: nil, dtype: :float32)
      TensorStream::Initializer.new(_op(:glorot_uniform, nil, nil, seed: seed, data_type: dtype))
    end

    def random_uniform_initializer(minval: 0, maxval: 1, seed: nil, dtype: nil)
      TensorStream::Initializer.new(_op(:random_uniform, nil, nil, minval: 0, maxval: 1, seed: seed, data_type: dtype))
    end

    def slice(input, start, size, name: nil)
      _op(:slice, input, start, size: size, name: name)
    end

    def zeros(shape, dtype: :float32, name: nil)
      _op(:zeros, shape, nil, data_type: dtype, name: name)
    end

    def ones(shape, dtype: :float32, name: nil)
      _op(:ones, shape, nil, data_type: dtype, name: name)
    end

    def less(input_a, input_b, name: nil)
      _op(:less, input_a, input_b, name: name)
    end

    def logical_and(input_a, input_b, name: nil)
      _op(:logical_and, input_a, input_b, name: name)
    end

    def greater(input_a, input_b, name: nil)
      _op(:greater, input_a, input_b, name: name)
    end

    def greater_equal(input_a, input_b, name: nil)
      _op(:greater_equal, input_a, input_b, name: name)
    end

    def less_equal(input_a, input_b, name: nil)
      _op(:less_equal, input_a, input_b, name: name)
    end

    def reduce_mean(input_tensor, axis = nil, keepdims: false, name: nil)
      _op(:reduce_mean, input_tensor, nil, axis: axis, keepdims: keepdims, name: name)
    end

    def reduce_sum(input_tensor, axis = nil, keepdims: false, name: nil)
      _op(:reduce_sum, input_tensor, nil, axis: axis, keepdims: keepdims, name: name)
    end

    def reduce_prod(input, axis = nil, keepdims: false, name: nil)
      _op(:reduce_prod, input, nil, axis: axis, keepdims: keepdims, name: name)
    end

    def concat(values, axis, name: 'concat')
      _op(:concat, values, nil, axis: axis, name: name)
    end

    def reshape(tensor, shape, name: nil)
      _op(:reshape, tensor, shape, name: name)
    end

    def square(tensor, name: nil)
      _op(:square, tensor, nil, name: name)
    end

    def cond(pred, true_fn, false_fn, name: nil)
      _op(:cond, true_fn, false_fn, pred: pred, name: name)
    end

    def where(condition, true_t = nil, false_t = nil, name: nil)
      _op(:where, true_t, false_t, pred: condition, name: name)
    end

    def add(input_a, input_b, name: nil)
      _op(:add, input_a, input_b, name: name)
    end

    def sub(input_a, input_b, name: nil)
      _op(:sub, input_a, input_b, name: name)
    end

    def max(input_a, input_b, name: nil)
      check_allowed_types(input_a, NUMERIC_TYPES)
      check_allowed_types(input_b, NUMERIC_TYPES)

      _op(:max, input_a, input_b, name: name)
    end

    def cast(input, dtype, name: nil)
      _op(:cast, input, nil, data_type: dtype, name: name)
    end

    def print(input, data, message: nil, name: nil)
      _op(:print, input, data, message: message, name: name)
    end

    def negate(input, options = {})
      _op(:negate, input, nil, options)
    end

    def equal(input_a, input_b, name: nil)
      _op(:equal, input_a, input_b, name: name)
    end

    def not_equal(input_a, input_b, name: nil)
      _op(:not_equal, input_a, input_b, name: name)
    end

    def zeros_like(tensor, dtype: nil, name: nil)
      _op(:zeros_like, tensor, nil, data_type: dtype, name: name)
    end

    def ones_like(tensor, dtype: nil, name: nil)
      _op(:ones_like, tensor, nil, data_type: dtype, name: name)
    end

    def identity(input, name: nil)
      _op(:identity, input, nil, name: name)
    end

    def multiply(input_a, input_b, name: nil)
      _op(:mul, input_a, input_b, name: name)
    end

    def pow(input_a, input_e, name: nil)
      _op(:pow, input_a, input_e, name: name)
    end

    def abs(input, name: nil)
      _op(:abs, input, nil, name: name)
    end

    def sign(input, name: nil)
      _op(:sign, input, nil, name: name)
    end

    def sin(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sin, input, nil, options)
    end

    def cos(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:cos, input, nil, options)
    end

    def tan(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:tan, input, nil, options)
    end

    def tanh(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:tanh, input, nil, options)
    end

    def sqrt(input, name: nil)
      options = {
        data_type: input.data_type,
        name: name
      }
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sqrt, input, nil, options)
    end

    def log(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:log, input, nil, options)
    end

    def exp(input, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:exp, input, nil, options)
    end

    def matmul(input_a, input_b, transpose_a: false,
               transpose_b: false,
               name: nil)
      _op(:matmul, input_a, input_b, transpose_a: transpose_a, transpose_b: transpose_b, name: name)
    end

    def transpose(tensor, perm: nil, name: 'transpose')
      _op(:transpose, tensor, nil, perm: perm, name: name)
    end

    def pad(tensor, paddings, mode: 'CONSTANT', name: nil)
      _op(:pad, tensor, nil, paddings: paddings, mode: mode, name: name)
    end
  end
end
