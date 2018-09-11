module TensorStream
  # TensorStream class that defines an operation
  class Operation < Tensor
    attr_accessor :name, :operation, :inputs, :rank, :options
    attr_reader :outputs

    def initialize(operation, *args)
      options = if args.last.is_a?(Hash)
                  args.pop
                else
                  {}
                end

      inputs = args

      setup_initial_state(options)

      @operation = operation
      @rank = options[:rank] || 0
      @name = [@graph.get_name_scope, options[:name] || set_name].compact.reject(&:empty?).join('/')
      @internal = options[:internal]
      @given_name = @name

      @options = options

      @inputs = inputs.map { |i| options[:preserve_params_type] ? i : TensorStream.convert_to_tensor(i) }
      @data_type = set_data_type(options[:data_type])
      @is_const = infer_const
      @shape = TensorShape.new(infer_shape)
      @graph.add_node(self)
    end

    def to_s
      @name
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(inputs)
      }
    end

    def infer_const
      return false if breakpoint
      case operation
      when :random_standard_normal, :random_uniform, :glorot_uniform, :print, :check_numerics
        false
      else
        non_const = @inputs.compact.find { |input| !input.is_const }
        non_const ? false : true
      end
    end

    def set_data_type(passed_data_type)
      case operation
      when :fill
        @inputs[1].data_type
      when :greater, :less, :equal, :not_equal, :greater_equal, :less_equal, :logical_and
        :boolean
      when :shape, :rank, :shape_n
        options[:out_type] || :int32
      when :random_standard_normal, :random_uniform, :glorot_uniform
        passed_data_type || :float32
      when :concat
        @inputs[1].data_type
      when :index
        if @inputs[0].is_a?(ControlFlow)

          if @inputs[1].is_const
            @inputs[0].inputs[@inputs[1].value].data_type
          else
            :unknown
          end
        else
          @inputs[0].data_type
        end
      else
        return passed_data_type if passed_data_type

        if @inputs[0]
          @inputs[0].data_type
        elsif @inputs[1]
          @inputs[1].data_type
        else
          :unknown
        end
      end
    end

    def to_math(name_only = false, max_depth = 99, cur_depth = 0)
      return @name if max_depth.zero?

      sub_input = auto_math(inputs[0], name_only, max_depth - 1, cur_depth + 1)
      sub_input2 = auto_math(inputs[1], name_only, max_depth - 1, cur_depth + 1) if inputs[1]

      out = case operation
            when :argmax
              "argmax(#{sub_input},#{options[:axis]})"
            when :negate
              "-#{sub_input}"
            when :index
              "#{sub_input}[#{sub_input2}]"
            when :slice
              "#{sub_input}[#{sub_input2}]"
            when :assign_sub
              "(#{inputs[0] ? inputs[0].name : 'self'} -= #{auto_math(inputs[1], name_only, 1)})"
            when :assign_add
              "(#{inputs[0] ? inputs[0].name : 'self'} += #{auto_math(inputs[1], name_only, 1)})"
            when :assign
              "(#{inputs[0] ? inputs[0].name : 'self'} = #{auto_math(inputs[1], name_only, 1)})"
            when :sin, :cos, :tanh
              "#{operation}(#{sub_input})"
            when :add
              "(#{sub_input} + #{sub_input2})"
            when :sub
              "(#{sub_input} - #{sub_input2})"
            when :pow
              "(#{sub_input}^#{sub_input2})"
            when :div
              "(#{sub_input} / #{sub_input2})"
            when :mul
              if auto_math(inputs[0]) == 1
                sub_input2
              elsif auto_math(inputs[1]) == 1
                sub_input
              else
                "(#{sub_input} * #{sub_input2})"
              end
            when :sum
              "sum(|#{sub_input}|,  axis=#{sub_input2})"
            when :mean
              "mean(|#{sub_input}|, axis=#{sub_input2})"
            when :prod
              "prod(|#{sub_input}|,  axis=#{sub_input2})"
            when :gradients
              "gradient(#{sub_input})"
            when :stop_gradient
              sub_input
            when :mat_mul
              "#{sub_input}.matmul(#{sub_input2})"
            when :eye
              "eye(#{sub_input})"
            when :transpose
              "transpose(#{sub_input})"
            when :shape
              "#{sub_input}.shape"
            when :exp
              "e^#{sub_input})"
            when :ones
              "ones(#{sub_input})"
            when :ones_like
              "ones_like(#{sub_input})"
            when :flow_group
              "flow_group(#{inputs.collect { |i| auto_math(i, name_only, max_depth - 1, cur_depth) }.join(',')})"
            when :zeros
              "zeros(#{sub_input})"
            when :reshape
              "reshape(#{sub_input},#{sub_input2})"
            when :rank
              "#{sub_input}.rank"
            when :cond
              "(#{auto_math(options[:pred], name_only, max_depth - 1, cur_depth)} ? #{sub_input} : #{sub_input2})"
            when :less
              "#{sub_input} < #{sub_input2}"
            when :less_equal
              "#{sub_input} <= #{sub_input2}"
            when :greater
              "#{sub_input} > #{sub_input2}"
            when :greater_equal
              "#{sub_input} >= #{sub_input2}"
            when :square
              "#{sub_input}\u00B2"
            when :log
              "log(#{sub_input})"
            when :identity
              "identity(#{sub_input})"
            when :print
              "print(#{sub_input})"
            when :pad
              "pad(#{sub_input},#{auto_math(options[:paddings])})"
            when :equal
              "#{sub_input} == #{sub_input2}"
            when :not_equal
              "#{sub_input} != #{sub_input2}"
            when :logical_and
              "#{sub_input} && #{sub_input2}"
            when :sqrt
              "sqrt(#{sub_input})"
            when :log1p
              "log1p(#{sub_input})"
            when :zeros_like
              "zeros_like(#{sub_input})"
            when :where
              "where(#{auto_math(options[:pred], name_only, max_depth - 1, cur_depth)}, #{sub_input}, #{sub_input2})"
            when :max
              "max(#{sub_input},#{sub_input2})"
            when :cast
              "cast(#{sub_input}, #{data_type})"
            when :broadcast_transform
              "broadcast_transform(#{sub_input},#{sub_input2})"
            when :broadcast_gradient_args
              "broadcast_transform(#{sub_input},#{sub_input2})"
            else
              "#{operation}(#{sub_input})" if sub_input
              "#{operation}(#{sub_input}, #{sub_input2})" if sub_input && sub_input2
            end
      ["\n", Array.new(cur_depth + 1) { ' ' }, out].flatten.join
    end

    def run
      eval
    end

    def op
      self
    end

    private

    def infer_shape
      case operation
      when :index
        input_shape = inputs[0].shape.shape
        return nil if input_shape.nil?
        input_shape[1, input_shape.size]
      when :mean, :prod, :sum
        return [] if inputs[1].nil?
        return nil if inputs[0].nil?
        input_shape = inputs[0].shape.shape
        return nil if input_shape.nil?
        return nil if inputs[1].is_a?(Tensor) && inputs[1].value.nil?

        axis = inputs[1].is_a?(Tensor) ? inputs[1].value : inputs[1]

        axis = [axis] unless axis.is_a?(Array)
        input_shape.each_with_index.map do |s, index|
          next nil if axis.include?(index)
          s
        end.compact
      when :reshape
        new_shape = inputs[1] && inputs[1].value ? inputs[1].value : nil
        return nil if new_shape.nil?
        return nil if inputs[0].shape.nil?

        input_shape = inputs[0].shape.shape
        return new_shape if input_shape.nil?
        return nil if input_shape.include?(nil)
        TensorShape.fix_inferred_elements(new_shape, input_shape.reduce(:*))
      when :flow_group
        []
      when :zeros, :ones, :fill
        inputs[0] ? inputs[0].value : options[:shape]
      when :zeros_like, :ones_like
        inputs[0].shape.shape
      when :shape
        inputs[0].shape.shape ? [inputs[0].shape.shape.size] : nil
      when :mat_mul
        shape1 = inputs[0].shape.shape.nil? ? nil : inputs[0].shape.shape[0]
        shape2 = inputs[1].shape.shape.nil? ? nil : inputs[1].shape.shape[1]
        [shape1, shape2]
      when :transpose
        return nil unless shape_full_specified(inputs[0])
        return nil if inputs[1].is_a?(Tensor)

        rank = inputs[0].shape.shape.size
        perm = inputs[1] || (0...rank).to_a.reverse
        perm.map { |p| inputs[0].shape.shape[p] }
      when :stack
        return nil unless shape_full_specified(inputs[0])

        axis = options[:axis] || 0
        new_shape = [inputs.size]
        inputs[0].shape.shape.inject(new_shape) { |ns, s| ns << s }
        rank = inputs[0].shape.shape.size + 1
        axis = rank + axis if axis < 0
        rotated_shape = Array.new(axis + 1) { new_shape.shift }
        rotated_shape.rotate! + new_shape
      when :concat
        return nil if inputs[0].value.nil?

        axis = inputs[0].value # get axis

        axis_size = 0

        inputs[1..inputs.size].each do |input_item|
          return nil if input_item.shape.shape.nil?
          return nil if input_item.shape.shape[axis].nil?

          axis_size += input_item.shape.shape[axis]
        end

        new_shape = inputs[1].shape.shape.dup
        new_shape[axis] = axis_size
        new_shape
      when :slice, :squeeze
        nil
      when :tile
        nil
      else
        return nil if inputs[0].nil?
        return inputs[0].shape.shape if inputs.size == 1
        TensorShape.infer_shape(inputs[0].shape.shape, inputs[1].shape.shape) if inputs.size == 2 && inputs[0] && inputs[1]
      end
    end

    def propagate_consumer(consumer)
      super
      @inputs.compact.each do |input|
        if input.is_a?(Array)
          input.flatten.compact.select { |t| t.is_a?(Tensor) }.each do |t|
            next if t.consumers.include?(consumer.name)
            t.send(:propagate_consumer, consumer)
          end
        elsif input.name != name && !input.consumers.include?(consumer.name)
          input.send(:propagate_consumer, consumer)
        end
      end
    end

    def propagate_outputs
      @inputs.compact.each do |input|
        if input.is_a?(Array)
          input.flatten.compact.each do |t|
            t.send(:setup_output, self) if t.is_a?(Tensor)
          end
        elsif input.is_a?(Tensor) && (input.name != name)
          input.send(:setup_output, self)
        end
      end
    end

    def set_name
      "#{@operation}#{graph.get_operation_counter}:#{@rank}"
    end
  end
end
