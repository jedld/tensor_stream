module TensorStream
  # Class that defines all available ops supported by TensorStream
  module Ops
    class OutputHolder
      def initialize(op)
        @op = op
      end
    end
    FLOATING_POINT_TYPES = %i[float32 float64 float].freeze
    INTEGER_TYPES = %i[uint8 int32 int int64].freeze
    NUMERIC_TYPES = FLOATING_POINT_TYPES + INTEGER_TYPES

    ##
    # Returns the index with the largest value across axes of a tensor.
    #
    # Argmuments
    #
    # +input+ A Tensor. Must be one of the following types: float32, float64, int32, int16
    # +axis+  Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0
    # +output_type+ Output data type defaults to int32
    def argmax(input, axis = nil, name: nil, dimension: nil, output_type: :int32)
      _op(:argmax, input, nil, axis: axis, name: name, dimension: dimension, data_type: output_type)
    end

    ##
    # Returns the index with the smallest value across axes of a tensor.
    #
    # Argmuments
    #
    # +input+ A Tensor. Must be one of the following types: float32, float64, int32, int16
    # +axis+  Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0
    # +output_type+ Output data type defaults to int32
    def argmin(input, axis = nil, name: nil, dimension: nil, output_type: :int32)
      _op(:argmin, input, nil, axis: axis, name: name, dimension: dimension, data_type: output_type)
    end

    ##
    # Constructs symbolic derivatives of ys of input w.r.t. x in wrt_xs.
    #
    # ys and xs are each a Tensor or a list of tensors. grad_ys is a list of Tensor, holding the gradients received by the ys. The list must be the same length as ys.
    #
    # Arguments:
    # +tensor_ys+ : A Tensor or list of tensors to be differentiated.
    # +wrt_xs+ : A Tensor or list of tensors to be used for differentiation.
    # +stop_gradients+ :  Optional. A Tensor or list of tensors not to differentiate through
    def gradients(tensor_ys, wrt_xs, name: 'gradients', stop_gradients: nil)
      gs = wrt_xs.collect do |x|
        stops = stop_gradients ? stop_gradients.map(&:name).join('_') : ''
        gradient_program_name = "grad_#{tensor_ys.name}_#{x.name}_#{stops}".to_sym
        tensor_graph = tensor_ys.graph

        tensor_program = if tensor_graph.node_added?(gradient_program_name)
                           tensor_graph.get_node(gradient_program_name)
                         else
                           tensor_graph.name_scope("gradient_wrt_#{x.name}") do
                             derivative_ops = TensorStream::MathGradients.derivative(tensor_ys, x, graph: tensor_graph,
                                                                                                   stop_gradients: stop_gradients)
                             tensor_graph.add_node!(gradient_program_name, derivative_ops)
                           end
                         end
        tensor_program
      end

      gs
    end

    ##
    # Outputs random values from a uniform distribution.
    def random_uniform(shape, dtype: :float32, minval: 0, maxval: 1, seed: nil, name: nil)
      options = { shape: shape, dtype: dtype, minval: minval, maxval: maxval, seed: seed, name: name }
      _op(:random_uniform, nil, nil, options)
    end

    ##
    # Outputs random values from a normal distribution.
    def random_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
      options = { shape: shape, dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name }
      _op(:random_standard_normal, nil, nil, options)
    end

    ##
    # Stops gradient computation.
    #
    # When executed in a graph, this op outputs its input tensor as-is.
    def stop_gradient(tensor, options = {})
      _op(:stop_gradient, tensor, nil, options)
    end

    ##
    # Construct an identity matrix
    def eye(num_rows, num_columns: nil, dtype: :float32, name: nil)
      _op(:eye, num_rows, num_columns || num_rows, data_type: dtype, name: name)
    end

    def expand_dims(input, axis = nil, name: nil)
      _op(:expand_dims, input, axis, name: name)
    end

    ##
    # This operation returns a 1-D integer tensor representing the shape of input
    def shape(input, name: nil, out_type: :int32)
      return constant(shape_eval(input, out_type), dtype: out_type, name: name) if input.is_a?(Array) && !input[0].is_a?(Tensor)
      return constant(input.shape.shape, dtype: out_type, name: "Shape/#{input.name}") if shape_full_specified(input)

      _op(:shape, input, name: name, out_type: out_type)
    end

    def shape_n(inputs, name: nil, out_type: :int32)
      shapes_known = true
      inputs.each do |input|
        unless input.shape.known?
          shapes_known = false
          break
        end
      end

      if shapes_known
        inputs.collect { |input| cons(input.shape.shape, dtype: out_type) }
      else
        res = _op(:shape_n, *inputs, out_type: out_type, name: name)
        Array.new(inputs.size) do |index|
          res[index]
        end
      end
    end

    ##
    # Constructs a tensor by tiling a given tensor.
    #
    # This operation creates a new tensor by replicating input multiples times.
    # The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,
    # and the values of input are replicated multiples[i] times along the 'i'th dimension. For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    def tile(input, multiples, name: nil)
      _op(:tile, input, multiples, name: name)
    end

    ##
    # Returns the rank of a tensor.
    def rank(input, name: nil)
      input = convert_to_tensor(input)
      return cons(input.shape.ndims) if input.shape.known?

      _op(:rank, input, name: name)
    end

    def constant_initializer(value, dtype: nil, verify_shape: false)
      TensorStream::Initializer.new(-> { convert_to_tensor(value, dtype: dtype) })
    end

    ##
    # initializer that generates tensors initialized to 0.
    #
    def zeros_initializer(dtype: :float32)
      TensorStream::Initializer.new(-> { _op(:zeros, nil, nil, data_type: dtype) })
    end

    ##
    # initializer that generates tensors initialized to 1.
    #
    def ones_initializer(dtype: :float32)
      TensorStream::Initializer.new(-> { _op(:ones, nil, nil, data_type: dtype) })
    end

    ##
    # The Glorot uniform initializer, also called Xavier uniform initializer.
    #
    # It draws samples from a uniform distribution within [-limit, limit]
    # where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number
    # of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    def glorot_uniform_initializer(seed: nil, dtype: nil)
      TensorStream::Initializer.new(-> { _op(:glorot_uniform, nil, nil, seed: seed, data_type: dtype) })
    end

    ##
    # Initializer that generates tensors with a uniform distribution.
    def random_uniform_initializer(minval: 0, maxval: 1, seed: nil, dtype: nil)
      TensorStream::Initializer.new(-> { _op(:random_uniform, nil, nil, minval: 0, maxval: 1, seed: seed, data_type: dtype) })
    end

    ##
    # Extracts a slice from a tensor.
    #
    # This operation extracts a slice of size size from a tensor input starting at the location specified by begin.
    # The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension of input that you want to slice. The starting location (begin) for the slice is
    # represented as an offset in each dimension of input. In other words, begin[i] is the offset into the 'i'th dimension of input that you want to slice from.
    def slice(input, start, size, name: nil)
      _op(:slice, input, start, size: size, name: name)
    end

    ##
    # Creates a tensor with all elements set to zero
    def zeros(shape, dtype: :float32, name: nil)
      _op(:zeros, shape, nil, data_type: dtype, name: name)
    end

    ##
    # Creates a tensor with all elements set to 1.
    def ones(shape, dtype: :float32, name: nil)
      _op(:ones, shape, nil, data_type: dtype, name: name)
    end

    ##
    # Returns element-wise largest integer not greater than x.
    def floor(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:floor, input, name: name)
    end

    ##
    # Returns element-wise smallest integer in not less than x
    def ceil(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:ceil, input, name: name)
    end

    ##
    # Returns the truth value of (x < y) element-wise.
    # This operation supports broadcasting
    def less(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:less, input_a, input_b, name: name)
    end

    ##
    # Returns the truth value of x AND y element-wise.
    def logical_and(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:logical_and, input_a, input_b, name: name)
    end

    ##
    # Returns the truth value of (x > y) element-wise.
    # This operation supports broadcasting
    def greater(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:greater, input_a, input_b, name: name)
    end

    ##
    # Returns the truth value of (x >= y) element-wise.
    #
    # This operation supports broadcasting
    def greater_equal(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:greater_equal, input_a, input_b, name: name)
    end

    ##
    # Returns the truth value of (x <= y) element-wise.
    def less_equal(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:less_equal, input_a, input_b, name: name)
    end

    ##
    # Computes the mean of elements across dimensions of a tensor.
    def reduce_mean(input_tensor, axis = nil, keepdims: false, name: nil)
      _op(:mean, input_tensor, axis, keepdims: keepdims, name: name)
    end

    ##
    # Computes the sum of elements across dimensions of a tensor.
    #
    # Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
    # the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
    # the reduced dimensions are retained with length 1.
    # If axis has no entries, all dimensions are reduced, and a tensor with a single element
    # is returned.
    def reduce_sum(input_tensor, axis = nil, keepdims: false, name: nil)
      _op(:sum, input_tensor, axis, keepdims: keepdims, name: name)
    end

    ##
    # Computes the product of elements across dimensions of a tensor.
    #
    # Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the
    # tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are
    # retained with length 1.
    #
    # If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
    def reduce_prod(input, axis = nil, keepdims: false, name: nil)
      _op(:prod, input, axis, keepdims: keepdims, name: name)
    end

    ##
    # Concatenates tensors along one dimension.
    def concat(values, axis, name: 'concat')
      if values.is_a?(Array)
        _op(:concat, axis, *values, name: name)
      else
        _op(:concat, axis, values, name: name)
      end
    end

    def split(value, num_or_size_splits, axis: 0, num: nil, name: 'split')
      value = convert_to_tensor(value)
      num_or_size_splits = convert_to_tensor(num_or_size_splits)
      axis = convert_to_tensor(axis)

      raise TensorStream::ValueError, "num_or_size_splits must be integer dtype" unless INTEGER_TYPES.include?(num_or_size_splits.data_type)

      res = _op(:split, value, num_or_size_splits, axis, name: name)

      pieces = if value.shape.known? && num_or_size_splits.is_const && num_or_size_splits.value && axis.is_const
                  if num_or_size_splits.shape.scalar?
                    raise TensorStream::ValueError, "num_or_size_splits must divide dimension #{value.shape.shape[axis.value]} evenly" unless value.shape.shape[axis.value] % num_or_size_splits.value == 0
                    div = num_or_size_splits.value
                    n = value.shape.shape[axis.value] / div

                    Array.new(div) { |i|
                      new_shape = value.shape.shape.dup
                      new_shape[axis.value] = n
                      new_shape
                    }
                  elsif num_or_size_splits.shape.ndims == 1
                    raise TensorStream::ValueError, "Sum of splits do not match total dimen in axis #{value.shape.shape[axis.value]} != #{ num_or_size_splits.value.reduce(:+)}" if value.shape.shape[axis.value] != num_or_size_splits.value.reduce(:+)
                    num_or_size_splits.value.collect do |v|
                      new_shape = value.shape.shape.dup
                      new_shape[axis.value] = v
                      new_shape
                    end
                  else
                    raise TensorStream::ValueError, "Scalar or 1D Tensor expected for num_or_size_splits"
                  end
                else
                  raise TensorStream::ValueError, "Cannot automatically determine num, please specify num: in options" if num.nil?

                  Array.new(num) { nil }
                end

      pieces.collect.with_index do |shape, i|
        op = index(res, i, name: "split/index:#{i}")
        if shape
          op.shape = TensorShape.new(shape)
        end
        op
      end
    end

    ##
    # select an index in an array or a set of tensor outputs
    def index(tensor, sel, name: nil)
      _op(:index, tensor, sel, name: name)
    end

    ##
    # Reshapes a tensor.
    #
    # Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
    def reshape(tensor, shape, name: nil)
      _op(:reshape, tensor, shape, name: name)
    end

    ##
    # Computes square of x element-wise.
    def square(tensor, name: nil)
      _op(:square, tensor, nil, name: name)
    end

    ##
    # Rounds the values of a tensor to the nearest integer, element-wise
    def round(tensor, name: nil)
      check_allowed_types(tensor, FLOATING_POINT_TYPES)
      _op(:round, tensor, nil, name: name)
    end

    ##
    # Computes the reciprocal of x element-wise.
    def reciprocal(tensor, name: nil)
      _op(:reciprocal, tensor, nil, name: name)
    end

    ##
    # Return true_fn() if the predicate pred is true else false_fn().
    def cond(pred, true_fn, false_fn, name: nil)
      _op(:cond, true_fn, false_fn, pred: pred, name: name)
    end

    ##
    # Return the elements, either from x or y, depending on the condition.
    def where(condition, true_t = nil, false_t = nil, name: nil)
      _op(:where, true_t, false_t, pred: condition, name: name)
    end

    ##
    # Returns x + y element-wise.
    #
    # This operation supports broadcasting
    def add(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:add, input_a, input_b, name: name)
    end

    ##
    # Adds all input tensors element-wise.
    #
    # Elements must all be the same shape and type
    def add_n(inputs, name: nil)
      _op(:add_n, *inputs, name: name)
    end

    ##
    # Computes asin of input element-wise
    def asin(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:asin, input, name: name)
    end

    ##
    # Computes acos of input element-wise
    def acos(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:acos, input, name: name)
    end

    ##
    # Computes atan of input element-wise
    def atan(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:atan, input, name: name)
    end

    ##
    # Returns x - y element-wise.
    #
    # This operation supports boradcasting
    def sub(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:sub, input_a, input_b, name: name)
    end

    ##
    # Returns element-wise remainder of division.
    def mod(input_a, input_b, name: nil)
      input_a = convert_to_tensor(input_a)
      input_b = convert_to_tensor(input_b)

      input_a, input_b = check_data_types(input_a, input_b)
       _op(:mod, input_a, input_b, name: name)
    end

    ##
    # Returns element-wise integer divistion.
    def floor_div(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:floor_div, input_a, input_b, name: name)
    end

    def range(start, limit, delta = 1, dtype: nil, name: 'range')
      _op(:range, start, limit, delta, data_type: dtype, name: name)
    end

    ##
    # Returns x - y element-wise.
    #
    # This operation supports boradcasting
    def subtract(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      sub(input_a, input_b, name: name)
    end

    ##
    # Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    def max(input_a, input_b, name: nil)
      check_allowed_types(input_a, NUMERIC_TYPES)
      check_allowed_types(input_b, NUMERIC_TYPES)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:max, input_a, input_b, name: name)
    end

    ##
    # Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    def maximum(input_a, input_b, name: nil)
      max(input_a, input_b, name: name)
    end

    ##
    # Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    def min(input_a, input_b, name: nil)
      check_allowed_types(input_a, NUMERIC_TYPES)
      check_allowed_types(input_b, NUMERIC_TYPES)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:min, input_a, input_b, name: name)
    end

    ##
    # Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    def minimum(input_a, input_b, name: nil)
      min(input_a, input_b, name: name)
    end

    ##
    # Casts a tensor to a new type, if needed
    def cast(input, dtype, name: nil)
      input = convert_to_tensor(input)
      return input if input.data_type == dtype

      _op(:cast, input, nil, data_type: dtype, name: name)
    end

    ##
    # Prints a list of tensors.
    #
    # This is an identity op (behaves like tf.identity) with the side effect of printing data when evaluating.
    def print(input, data, message: nil, name: nil)
      _op(:print, input, data, message: message, name: name)
    end

    ##
    # Computes numerical negative value element-wise.
    def negate(input, name: nil)
      _op(:negate, input, nil, name: name)
    end

    ##
    # Computes numerical negative value element-wise.
    def negative(input, name: nil)
      negate(input, name: name)
    end

    ##
    # Returns the truth value of (x == y) element-wise.
    def equal(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:equal, input_a, input_b, name: name)
    end

    ##
    # Returns the truth value of (x != y) element-wise.
    # This ops supports broadcasting
    def not_equal(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:not_equal, input_a, input_b, name: name)
    end

    ##
    # reates a tensor with all elements set to zero.
    # Given a single tensor (tensor), this operation returns a tensor
    # of the same type and shape as tensor with all elements set to zero.
    # Optionally, you can use dtype to specify a new type for the returned tensor.
    def zeros_like(tensor, dtype: nil, name: nil)
      _op(:zeros_like, tensor, nil, data_type: dtype, name: name)
    end

    ##
    # Creates a tensor with all elements set to 1.
    # Given a single tensor (tensor), this operation returns a
    # tensor of the same type and shape as tensor with all elements set to 1.
    # Optionally, you can specify a new type (dtype) for the returned tensor.
    def ones_like(tensor, dtype: nil, name: nil)
      _op(:ones_like, tensor, nil, data_type: dtype, name: name)
    end

    ##
    # Return a tensor with the same shape and contents as input.
    def identity(input, name: nil)
      _op(:identity, input, nil, name: name)
    end

    ##
    # Returns x * y element-wise.
    # This operation supports broadcasting
    def multiply(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:mul, input_a, input_b, name: name)
    end

    ##
    # Returns x * y element-wise.
    # This operation supports broadcasting
    def mul(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:mul, input_a, input_b, name: name)
    end

    ##
    # Divides x / y elementwise
    # This operation supports broadcasting
    def div(input_a, input_b, name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:div, input_a, input_b, name: name)
    end

    ##
    # Computes the power of one value to another.
    def pow(input_a, input_e, name: nil)
      input_a, input_e = check_data_types(input_a, input_e)
      _op(:pow, input_a, input_e, name: name)
    end

    ##
    # Computes the absolute value of a tensor.
    def abs(input, name: nil)
      _op(:abs, input, nil, name: name)
    end

    ##
    # Returns an element-wise indication of the sign of a number.
    # y = sign(x) = -1 if x < 0; 0 if x == 0 or tf.is_nan(x); 1 if x > 0.
    # Zero is returned for NaN inputs.
    def sign(input, name: nil)
      _op(:sign, input, nil, name: name)
    end

    ##
    # Computes sin of input element-wise.
    def sin(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sin, input, nil, name: name)
    end

    ##
    # Computes cos of input element-wise.
    def cos(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:cos, input, nil, name: name)
    end

    ##
    # Computes tan of input element-wise.
    def tan(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:tan, input, nil, name: name)
    end

    ##
    # Computes tanh of input element-wise.
    def tanh(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:tanh, input, nil, name: name)
    end

    ##
    # Computes sqrt of input element-wise.
    def sqrt(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sqrt, input, nil, name: name)
    end

    ##
    # Computes natural logarithm of x element-wise.
    def log(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:log, input, nil, name: name)
    end

    ##
    # Computes natural logarithm of (1 + x) element-wise.
    def log1p(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:log1p, input, nil, name: name)
    end

    ##
    # Computes exponential of x element-wise.
    def exp(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:exp, input, nil, name: name)
    end

    ##
    # Creates a tensor filled with a scalar value.
    #
    # This operation creates a tensor of shape dims and fills it with value.
    #
    # For example:
    # Output tensor has shape [2, 3].
    # fill([2, 3], 9) => [[9, 9, 9]
    #                    [9, 9, 9]]
    def fill(dims, value, name: nil)
      _op(:fill, dims, value, name: name)
    end

    ##
    # Computes sigmoid of x element-wise.
    def sigmoid(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sigmoid, input, nil, name: name)
    end

    ##
    # Multiplies matrix a by matrix b, producing a * b.
    # The inputs must, following any transpositions, be tensors of rank 2 .
    def matmul(input_a, input_b, transpose_a: false,
               transpose_b: false,
               name: nil)
      input_a, input_b = check_data_types(input_a, input_b)
      _op(:mat_mul, input_a, input_b, transpose_a: transpose_a, transpose_b: transpose_b, name: name)
    end

    ##
    # Transposes a. Permutes the dimensions according to perm.
    def transpose(tensor, perm = nil, name: 'transpose')
      _op(:transpose, tensor, perm, name: name)
    end

    ##
    # Pads a tensor.
    # This operation pads a tensor according to the paddings you specify.
    def pad(tensor, paddings, mode: 'CONSTANT', name: nil)
      _op(:pad, tensor, nil, paddings: paddings, mode: mode, name: name)
    end

    ##
    # Checks a tensor for NaN and Inf values.
    # When run, reports an InvalidArgument error if tensor has any values that are not a number (NaN) or infinity (Inf). Otherwise, passes tensor as-is.
    def check_numerics(tensor, message, name: nil)
      _op(:check_numerics, tensor, nil, message: message, name: name)
    end

    def size(tensor, name: nil, out_type: :int32)
      _op(:size, tensor, name: name, out_type: out_type)
    end

    def squared_difference(input_a, input_b, name: nil)
      _op(:squared_difference, input_a, input_b, name: name)
    end

    def broadcast_gradient_args(shape_a, shape_b, name: nil)
      op_result = _op(:broadcast_gradient_args, shape_a, shape_b, name: name)
      [op_result[0], op_result[1]]
    end

    ##
    # Gather slices from params and axis according to indices.
    #
    def gather(params, indices, validate_indices: nil,
                                name: nil,
                                axis: 0)
      _op(:gather, params, indices, validate_indices: validate_indices, name: name, axis: axis)
    end


    ##
    # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
    #
    def stack(values, axis: 0, name: 'stack')
      _op(:stack, *values, axis: axis, name: name)
    end

    ##
    # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
    #
    def unstack(value, num: nil, axis: 0, name: 'unstack')
      res = _op(:unstack, value, num: num, axis: axis, name: name)

      num_vars = if value.shape.known?
                   new_shape = value.shape.shape.dup
                   rank = new_shape.size - 1
                   axis = rank + axis if axis < 0
                   rotated_shape = Array.new(axis + 1) { new_shape.shift }
                   new_shape = rotated_shape.rotate!(-1) + new_shape
                   new_shape[0]
                 else
                   raise TensorStream::ValueError, "num is unspecified and cannot be inferred." if num.nil?
                   num
                 end

      return res[0] if num_vars == 1

      Array.new(num_vars) do |i|
        index(res, i, name: "unstack/index:#{i}")
      end
    end

    ##
    # Same as stack
    def pack(values, axis: 0, name: 'pack')
      _op(:stack, *values, axis: axis, name: name)
    end

    ##
    # Same as unstack
    #
    def unpack(value, num: nil, axis: 0, name: 'unpack')
      unstack(value, num: num, axis: axis, name: name)
    end

    ##
    # Removes dimensions of size 1 from the shape of a tensor.
    #
    # Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed.
    # If you don't want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying axis.
    def squeeze(value, axis: [], name: nil)
      _op(:squeeze, value, axis: axis, name: nil)
    end

    ##
    # Computes the difference between two lists of numbers or strings.
    # Given a list x and a list y, this operation returns a list out that represents all values
    # that are in x but not in y. The returned list out is sorted in the same order that the numbers appear
    # in x (duplicates are preserved). This operation also returns a list idx that represents the position of
    # each out element in x. In other words:
    #
    def setdiff1d(x, y, index_dtype: :int32, name: nil)
      result = _op(:setdiff1d, x, y, index_dtype: index_dtype, name: name)
      [result[0], result[1]]
    end

    def cumprod(x, axis: 0, exclusive: false, reverse: false, name: nil)
      _op(:cumprod, x, axis: axis, exclusive: exclusive, reverse: reverse, name: name)
    end

    def invert_permutation(x, name: nil)
      _op(:invert_permutation, x, name: name)
    end
  end
end
