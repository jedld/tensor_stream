
module TensorStream
  # Class that defines all available ops supported by TensorStream
  module Ops
    if File.exists?(File.join(__dir__, 'generated_stub', 'ops.rb'))
      require 'tensor_stream/generated_stub/ops'
      include TensorStream::OpStub
    end

    class OutputHolder
      def initialize(op)
        @op = op
      end
    end
    FLOATING_POINT_TYPES = %i[float32 float64 float float16].freeze
    INTEGER_TYPES = %i[uint8 int32 int int16 uint16 int64 uint32 uint64].freeze
    NUMERIC_TYPES = FLOATING_POINT_TYPES + INTEGER_TYPES

    ##
    # Assert the condition x == y holds element-wise.
    #
    # Argmuments
    #
    # +x+ Numeric Tensor.
    # +y+ Numeric Tensor, same dtype as and broadcastable to x.
    #
    # Returns
    # Op that raises InvalidArgumentError if x == y is false
    def assert_equal(x, y, data: nil, summarize: nil, message: nil, name: nil)
      _op(:assert_equal, x, y, data: data, summarize: summarize, message: message, name: name)
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
    def gradients(tensor_ys, wrt_xs, name: "gradients", stop_gradients: nil)
      tensor_ys = tensor_ys.op
      gs = wrt_xs.map(&:op).collect { |x|
        stops = stop_gradients ? stop_gradients.map(&:name).join("_") : ""
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
      }

      gs
    end

    ##
    # Outputs random values from a normal distribution.
    def random_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
      options = {dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name}
      _op(:random_standard_normal, shape, options)
    end

    ##
    # Outputs random values from a truncated normal distribution.
    def truncated_normal(shape, dtype: :float32, mean: 0.0, stddev: 1.0, seed: nil, name: nil)
      options = {dtype: dtype, mean: mean, stddev: stddev, seed: seed, name: name}
      _op(:truncated_normal, shape, options)
    end

    ##
    # Stops gradient computation.
    #
    # When executed in a graph, this op outputs its input tensor as-is.
    def stop_gradient(tensor, options = {})
      _op(:stop_gradient, tensor, options)
    end

    ##
    # Construct an identity matrix
    def eye(num_rows, num_columns: nil, dtype: :float32, name: nil)
      _op(:eye, num_rows, num_columns || num_rows, data_type: dtype, name: name)
    end

    def expand_dims(input, axis = nil, name: nil)
      _op(:expand_dims, input, axis, name: name)
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
        inputs.collect { |input| cons(input.shape.shape, dtype: out_type).op }
      else
        res = _op(:shape_n, *inputs, out_type: out_type, name: name)
        Array.new(inputs.size) do |index|
          res[index]
        end
      end
    end

    ##
    # initializer that generates tensors initialized to 0.
    #
    def zeros_initializer(dtype: :float32)
      TensorStream::Initializer.new(-> { _op(:zeros, data_type: dtype) })
    end

    ##
    # initializer that generates tensors initialized to 1.
    #
    def ones_initializer(dtype: :float32)
      TensorStream::Initializer.new(-> { _op(:ones, data_type: dtype) })
    end

    def constant_initializer(value, dtype: nil, verify_shape: false)
      TensorStream::Initializer.new(-> { _op(:fill, nil, convert_to_tensor(value, dtype: dtype)) })
    end

    ##
    # The Glorot uniform initializer, also called Xavier uniform initializer.
    #
    # It draws samples from a uniform distribution within [-limit, limit]
    # where limit is sqrt(6 / (fan_in + fan_out)) where fan_in is the number
    # of input units in the weight tensor and fan_out is the number of output units in the weight tensor.
    def glorot_uniform_initializer(seed: nil, dtype: nil)
      TensorStream::Initializer.new(-> { _op(:glorot_uniform, seed: seed, data_type: dtype) })
    end

    ##
    # Initializer that generates tensors with a uniform distribution.
    def random_uniform_initializer(minval: 0, maxval: 1, seed: nil, dtype: nil)
      TensorStream::Initializer.new(-> { _op(:random_uniform, minval: 0, maxval: 1, seed: seed, data_type: dtype) })
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
    # Creates a tensor with all elements set to 1.
    def ones(shape, dtype: :float32, name: nil)
      _op(:ones, shape, data_type: dtype, name: name)
    end

    ##
    # Returns the truth value of x AND y element-wise.
    def logical_and(input_a, input_b, name: nil)
      check_data_types(input_a, input_b)
      _op(:logical_and, input_a, input_b, name: name)
    end

    ##
    # Computes the mean of elements across dimensions of a tensor.
    def reduce_mean(input_tensor, axis = nil, keepdims: false, name: nil)
      reduce(:mean, input_tensor, axis, keepdims: keepdims, name: name)
    end

    def reduce(op, input, axis = nil, keepdims: false, name: nil)
      input = TensorStream.convert_to_tensor(input)
      return input if input.shape.scalar?

      axis = cast_axis(input, axis)

      _op(op, input, axis, keepdims: keepdims, name: name)
    end

    ##
    # Concatenates tensors along one dimension.
    def concat(values, paxis = nil, axis: nil, name: "concat")
      if values.is_a?(Array)
        _op(:concat, paxis || axis, *values, name: name)
      else
        _op(:concat, paxis || axis, values, name: name)
      end
    end

    ##
    # Partitions data into num_partitions tensors using indices from partitions
    def dynamic_partition(data, partitions, num_partitions, name: nil)
      result = _op(:dynamic_partition, data, partitions, num_partitions: num_partitions, name: nil)
      Array.new(num_partitions) do |index|
        result[index]
      end
    end

    def split(value, num_or_size_splits, axis: 0, num: nil, name: "split")
      value = convert_to_tensor(value)
      num_or_size_splits = convert_to_tensor(num_or_size_splits)
      axis = convert_to_tensor(axis)

      raise TensorStream::ValueError, "num_or_size_splits must be integer dtype" unless INTEGER_TYPES.include?(num_or_size_splits.data_type)

      res = _op(:split, value, num_or_size_splits, axis, name: name)

      pieces = if value.shape.known? && num_or_size_splits.is_const && num_or_size_splits.value && axis.is_const
        if num_or_size_splits.shape.scalar?
          raise TensorStream::ValueError, "num_or_size_splits must divide dimension #{value.shape.shape[axis.value]} evenly" unless (value.shape.shape[axis.value] % num_or_size_splits.value).zero?

          div = num_or_size_splits.value
          n = value.shape.shape[axis.value] / div

          Array.new(div) do
            new_shape = value.shape.shape.dup
            new_shape[axis.value] = n
            new_shape
          end
        elsif num_or_size_splits.shape.ndims == 1
          raise TensorStream::ValueError, "Sum of splits do not match total dimen in axis #{value.shape.shape[axis.value]} != #{num_or_size_splits.value.reduce(:+)}" if value.shape.shape[axis.value] != num_or_size_splits.value.reduce(:+)

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
        op.set_shape(shape) if shape

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
      _op(:square, tensor, name: name)
    end

    ##
    # Computes the reciprocal of x element-wise.
    def reciprocal(tensor, name: nil)
      _op(:reciprocal, tensor, name: name)
    end

    ##
    # Return true_fn() if the predicate pred is true else false_fn().
    def cond(pred, true_fn, false_fn, name: nil)
      _op(:case, [pred], false_fn, true_fn, name: name)
    end

    ##
    # Return the elements, either from x or y, depending on the condition.
    def where(condition, true_t = nil, false_t = nil, name: nil)
      _op(:where, condition, true_t, false_t, name: name)
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
    # Returns element-wise integer divistion.
    def floor_div(input_a, input_b, name: nil)
      check_data_types(input_a, input_b)
      _op(:floor_div, input_a, input_b, name: name)
    end

    ##
    # Casts a tensor to a new type, if needed
    def cast(input, dtype, name: nil)
      input = convert_to_tensor(input)
      return input if input.data_type == dtype

      _op(:cast, input, data_type: dtype, name: name)
    end

    ##
    # Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    def maximum(input_a, input_b, name: nil)
      max(input_a, input_b, name: name)
    end

    ##
    # Returns the min of x and y (i.e. x < y ? x : y) element-wise.
    def minimum(input_a, input_b, name: nil)
      min(input_a, input_b, name: name)
    end

    ##
    # Prints a list of tensors.
    #
    # This is an identity op (behaves like tf.identity) with the side effect of printing data when evaluating.
    def print(input, data, message: nil, name: nil)
      _op(:print, input, data, message: message, name: name)
    end

    ##
    # Returns the truth value of (x != y) element-wise.
    # This ops supports broadcasting
    def not_equal(input_a, input_b, name: nil)
      check_data_types(input_a, input_b)
      _op(:not_equal, input_a, input_b, name: name)
    end

    ##
    # reates a tensor with all elements set to zero.
    # Given a single tensor (tensor), this operation returns a tensor
    # of the same type and shape as tensor with all elements set to zero.
    # Optionally, you can use dtype to specify a new type for the returned tensor.
    def zeros_like(tensor, dtype: nil, name: nil)
      _op(:zeros_like, tensor, data_type: dtype, name: name)
    end

    ##
    # Creates a tensor with all elements set to 1.
    # Given a single tensor (tensor), this operation returns a
    # tensor of the same type and shape as tensor with all elements set to 1.
    # Optionally, you can specify a new type (dtype) for the returned tensor.
    def ones_like(tensor, dtype: nil, name: nil)
      _op(:ones_like, tensor, data_type: dtype, name: name)
    end

    ##
    # Return a tensor with the same shape and contents as input.
    def identity(input, name: nil)
      _op(:identity, input, name: name)
    end

    ##
    # Returns x * y element-wise.
    # This operation supports broadcasting
    def multiply(input_a, input_b, name: nil)
      check_data_types(input_a, input_b)
      _op(:mul, input_a, input_b, name: name)
    end

    ##
    # Computes the absolute value of a tensor.
    def abs(input, name: nil)
      _op(:abs, input, name: name)
    end

    ##
    # Computes sec of input element-wise.
    def sec(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sec, input, name: name)
    end

    ##
    # Computes sqrt of input element-wise.
    def sqrt(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:sqrt, input, name: name)
    end

    ##
    # Computes natural logarithm of x element-wise.
    def log(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:log, input, name: name)
    end

    ##
    # Computes natural logarithm of (1 + x) element-wise.
    def log1p(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:log1p, input, name: name)
    end

    ##
    # Computes exponential of x element-wise.
    def exp(input, name: nil)
      check_allowed_types(input, FLOATING_POINT_TYPES)
      _op(:exp, input, name: name)
    end

    ##
    # Transposes a. Permutes the dimensions according to perm.
    def transpose(tensor, perm = nil, name: "transpose")
      _op(:transpose, tensor, perm, name: name)
    end

    ##
    # Pads a tensor.
    # This operation pads a tensor according to the paddings you specify.
    def pad(tensor, paddings, mode: "CONSTANT", name: nil)
      _op(:pad, tensor, paddings, mode: mode, name: name)
    end

    ##
    # Checks a tensor for NaN and Inf values.
    # When run, reports an InvalidArgument error if tensor has any values that are not a number (NaN) or infinity (Inf). Otherwise, passes tensor as-is.
    def check_numerics(tensor, message, name: nil)
      _op(:check_numerics, tensor, message: message, name: name)
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
    def stack(values, axis: 0, name: "stack")
      _op(:stack, *values, axis: axis, name: name)
    end

    ##
    # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
    #
    def unstack(value, num: nil, axis: 0, name: "unstack")
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
    def pack(values, axis: 0, name: "pack")
      _op(:stack, *values, axis: axis, name: name)
    end

    ##
    # Same as unstack
    #
    def unpack(value, num: nil, axis: 0, name: "unpack")
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

    def clip_by_norm(tensor, clip_norm, axes: nil, name: nil)
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

    ##
    # Create a case operation.
    #
    # The pred_fn_pairs parameter is a dict or list of pairs of size N.
    # Each pair contains a boolean scalar tensor and a proc that creates the tensors to be returned if the boolean evaluates to true.
    # default is a proc generating a list of tensors. All the proc in pred_fn_pairs as well as default (if provided) should return the
    # same number and types of tensors.
    #
    def case(args = {})
      args = args.dup
      default = args.delete(:default)
      exclusive = args.delete(:exclusive)
      strict = args.delete(:strict)
      name = args.delete(:name)

      predicates = []
      functions = []

      args.each do |k, v|
        raise "Invalid argment or option #{k}" unless k.is_a?(Tensor)

        predicates << k
        functions << (v.is_a?(Proc) ? v.call : v)
      end

      _op(:case, predicates, default, *functions, exclusive: exclusive, strict: strict, name: name)
    end

    def cumprod(x, axis: 0, exclusive: false, reverse: false, name: nil)
      _op(:cumprod, x, axis: axis, exclusive: exclusive, reverse: reverse, name: name)
    end

    def invert_permutation(x, name: nil)
      _op(:invert_permutation, x, name: name)
    end

    def cast_axis(input, axis)
      if !axis.nil?
        axis
      elsif input.shape.known?
        (0...input.shape.ndims).to_a
      else
        range(0, rank(input))
      end
    end
  end
end
