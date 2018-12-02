
module TensorStream
  # module that contains helper functions useful for ops
  module OpHelper
    def _op(code, *args)
      default_graph = Graph.get_default_graph

      op = default_graph.add_op(code.to_sym, *args)
      if !default_graph.get_dependency_scope.nil?
        i_op(:identity, op, default_graph.get_dependency_scope, name: [op.name, 'tuple', 'control_dependency'].join('/'))
      else
        op
      end
    end

    # same as op but with a marker that it was internal generated
    def i_op(code, *args)
      options = if args.last.is_a?(Hash)
                  args.pop
                else
                  {}
                end

      args << options.merge(internal: true)
      Graph.get_default_graph.add_op(code.to_sym, *args)
    end

    def cons(value, options = {})
      TensorStream.constant(value, options)
    end

    def i_cons(value, options = {})
      TensorStream.constant(value, options.merge(internal: true))
    end

    def shape_eval(input, output_type = :int32)
      return [] unless input.is_a?(Array)
      arr = []
      arr_ptr = input

      Kernel.loop do
        arr << (TensorStream::Ops::FLOATING_POINT_TYPES.include?(output_type) ? arr_ptr.size.to_f : arr_ptr.size)
        arr_ptr = arr_ptr[0]

        break unless arr_ptr.is_a?(Array)
      end

      arr
    end

    def fp_type?(type)
      TensorStream::Ops::FLOATING_POINT_TYPES.include?(type)
    end

    def int_type?(type)
      TensorStream::Ops::INTEGER_TYPES.include?(type)
    end

    def format_source(trace)
      grad_source = trace.select { |c| c.to_s.include?(File.join('lib', 'tensor_stream', 'math_gradients')) }.first
      source = trace.reject { |c| c.to_s.include?(File.join('lib', 'tensor_stream')) }.first
      [grad_source, trace].compact.join("\n")
    end

    def shapes_fully_specified_and_equal(x, y)
      return false if !shape_full_specified(x) || !shape_full_specified(y)
      return false if x.shape.shape != y.shape.shape

      true
    end

    def shape_full_specified(tensor)
      return false if tensor.shape.nil?
      return false if tensor.shape.shape.nil?

      tensor.shape.shape.each { |s| return false if s.nil? || (s < 0) }
      true
    end

    def reduced_shape(input_shape, axes)
      input_shape = TensorStream.convert_to_tensor(input_shape)
      axes = TensorStream.convert_to_tensor(axes)
      input_rank = i_op(:size, input_shape)
      axes = TensorStream.range(0, input_rank) if axes.nil?
      axes = (axes + input_rank) % input_rank
      axes_shape = i_op(:shape, axes)

      TensorStream.dynamic_stitch([TensorStream.range(0, input_rank), axes],
                                  [input_shape, i_op(:fill, axes_shape, 1)])
    end
  end
end
