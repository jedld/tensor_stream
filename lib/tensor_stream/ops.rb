module TensorStream
  module Ops
    FLOATING_POINT_TYPES = %w[float32 float64].map(&:to_sym)
    NUMERIC_TYPES = %w[int32 int64 float32 float64].map(&:to_sym)

    def argmax(input, axis = nil, name: nil, dimension: nil, output_type: :int32)
      op(:argmax, input, nil, axis: axis, name: name, dimension: dimension, data_type: output_type)
    end

    def gradients(ys, xs, grad_ys: nil,
      name: 'gradients',
      colocate_gradients_with_ops: false,
      gate_gradients: false,
      aggregation_method: nil,
      stop_gradients: nil
      )
      options = { stop_gradients: stop_gradients}
      op(:gradients, ys, xs, options)
    end

    def random_uniform(shape, dtype: :float32, minval: 0, maxval: 1, seed: nil, name: nil)
      options = {shape: shape, dtype: dtype, minval: minval, maxval: maxval, seed: seed, name: name}
      op(:random_uniform, nil, nil, options)
    end

    def random_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
      options = {shape: shape, dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name}
      op(:random_normal, nil, nil, options)
    end

    def stop_gradient(tensor, options = {})
      op(:stop_gradient, tensor, nil, options)
    end

    def eye(num_rows, num_columns: nil, dtype: :float32, name: nil)
      op(:eye, num_rows, num_columns || num_rows, data_type: dtype, name: name, preserve_params_type: true)
    end

    def shape(input, name: nil, out_type: :int32)
      op(:shape, input, nil, name: name)
    end

    def rank(input, name: nil)
      op(:rank, input, name: name)
    end

    def zeros_initializer(options = {})
      op(:zeros, nil, nil, options)
    end
  
    def slice(input, start, size, name: nil)
      op(:slice, input, start, size: size, name: name)
    end
  
    def zeros(shape, dtype: :float32, name: nil)
      op(:zeros, shape, nil, data_type: dtype, name: name)
    end
  
    def ones(shape, dtype: :float32, name: nil)
      op(:ones, shape, nil, data_type: dtype, name: name)
    end
  
    def less(a, b, name: nil)
      op(:less, a, b, name: name)
    end
  
    def greater(a, b, name: nil)
      op(:greater, a, b, name: name)
    end
  
    def reduce_mean(input_tensor, axis = nil, keepdims: false, name: nil)
      op(:reduce_mean, input_tensor, nil, axis: axis, keepdims: keepdims, name: name)
    end

    def reduce_sum(input_tensor, axis = nil, keepdims: false, name: nil)
      op(:reduce_sum, input_tensor, nil, axis: axis, keepdims: keepdims, name: name)
    end
  
    def reduce_prod(input, axis = nil, keepdims: false, name: nil)
      op(:reduce_prod, input, nil, axis: axis, keepdims: keepdims, name: name)
    end
  
    def concat(values, axis, name: 'concat')
      op(:concat, values, nil, axis: axis, name: name)
    end
  
    def reshape(tensor, shape, name: nil)
      op(:reshape, tensor, shape, name: name)
    end
  
    def square(tensor, name: nil)
      op(:square, tensor, nil, name: name)
    end
  
    def cond(pred, true_fn, false_fn, name: nil)
      op(:cond, true_fn, false_fn, pred: pred, name: name)
    end

    def where(condition, x = nil, y = nil, name: nil)
      op(:where, x, y, pred: condition)
    end
  
    def add(a, b, name: nil)
      op(:add, a, b, name: name)
    end
  
    def sub(a, b, name: nil)
      op(:sub, a, b, name: name)
    end

    def max(a, b, name: nil)
      check_allowed_types(a, NUMERIC_TYPES)
      check_allowed_types(b, NUMERIC_TYPES)

      op(:max, a, b, name: name)
    end

    def cast(input, dtype, name: nil)
      op(:cast, input, nil, data_type: dtype, name: name)
    end
  
    def print(input, data, message: nil, name: nil)
      op(:print, input, data, message: message, name: name)
    end
  
    def negate(a, options = {})
      op(:negate, a, nil, options)
    end
  
    def equal(a, b, name: nil)
      op(:equal, a, b, name: name)
    end

    def not_equal(a, b, name: nil)
      op(:not_equal, a, b, name: name)
    end
  
    def zeros_like(tensor, dtype: nil, name: nil)
      op(:zeros_like, tensor, nil, data_type: dtype, name: name)
    end

    def ones_like(tensor, dtype: nil, name: nil)
      op(:ones_like, tensor, nil, data_type: dtype, name: name)
    end
  
    def identity(input, name: nil)
      op(:identity, input, nil, name: name)
    end
  
    def multiply(a, b, name: nil)
      op(:mul, a, b, name: name)
    end
  
    def pow(a, e, name: nil)
      op(:pow, a, e, name: name)
    end
  
    def abs(x, name: nil)
      op(:abs, x, nil, name: name)
    end
  
    def sign(x, name: nil)
      op(:sign, x, nil, name: name)
    end
  
    def sin(a, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:sin, a, nil, options)
    end
  
    def cos(a, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:cos, a, nil, options)
    end
  
    def tan(a, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:tan, a, nil, options)
    end
  
    def tanh(a, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:tanh, a, nil, options)
    end
  
    def sqrt(a, name: nil)
      options = {
        data_type: a.data_type,
        name: name
      }
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:sqrt, a, nil, options)
    end
  
    def log(input, options= {})
      options[:data_type] ||= :float32
      check_allowed_types(input, FLOATING_POINT_TYPES)
      op(:log, input, nil, options)
    end
  
    def exp(a, options = {})
      options[:data_type] ||= :float32
      check_allowed_types(a, FLOATING_POINT_TYPES)
      op(:exp, a, nil, options)
    end

    def matmul(input_a, input_b, transpose_a: false,
                                 transpose_b: false,
                                 name: nil)
      op(:matmul, input_a, input_b, transpose_a: transpose_a, transpose_b: transpose_b, name: name)
    end

    def transpose(tensor, perm: nil, name: 'transpose')
      op(:transpose, tensor, nil, perm: perm, name: name)
    end

    def pad(tensor, paddings, mode: 'CONSTANT', name: nil)
      op(:pad, tensor, nil, paddings: paddings, mode: mode, name: name)
    end
  end
end