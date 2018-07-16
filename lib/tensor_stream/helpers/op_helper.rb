module TensorStream
  # module that contains helper functions useful for ops
  module OpHelper
    def _op(code, t_a, t_b = nil, options = {})
      op = Operation.new(code.to_sym, t_a, t_b, options)
      if !TensorStream.get_default_graph.get_dependency_scope.nil?
        i_op(:identity, op, TensorStream.get_default_graph.get_dependency_scope, name: [op.name, 'tuple', 'control_dependency'].join('/'))
      else
        op
      end
    end

    # same as op but with a marker that it was internal generated
    def i_op(code, t_a, t_b = nil, options = {})
      Operation.new(code.to_sym, t_a, t_b, options.merge(internal: true))
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

    def dtype_eval(rank, value, data_type = nil)
      dtype = if data_type.nil?
        Tensor.detect_type(value[0])
      else
       data_type
      end

      rank += 1 if dtype == :array

      [dtype, rank, value[0], value.size]
    end

    def val_to_dtype(value)
      if value.is_a?(String)
        :string
      elsif value.is_a?(Float)
        :float32
      elsif value.is_a?(Integer)
        :int32
      elsif value.is_a?(Array)
        :array
      else
        :float32
      end
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
      [grad_source, source].compact.join("\n")
    end

    def shapes_fully_specified_and_equal(x, y)
      return false if !shape_full_specified(x) || !shape_full_specified(y)
      return false if x.shape.shape != y.shape.shape
      
      true
     end
 
     def shape_full_specified(tensor)
       return false if tensor.shape.nil?
       return false if tensor.shape.shape.nil?
 
       tensor.shape.shape.each { |s| return false if s.nil? }
       true
     end
  end
end
