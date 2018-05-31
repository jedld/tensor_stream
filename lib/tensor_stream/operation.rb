module TensorStream
  # TensorStream class that defines an operation
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options
    attr_reader :outputs

    def initialize(operation, input_a, input_b, options = {})
      setup_initial_state(options)

      @operation = operation
      @rank = options[:rank] || 0
      @name = [@graph.get_name_scope, options[:name] || set_name].compact.reject(&:empty?).join('/')
      @internal = options[:internal]
      @given_name = @name

      @options = options

      @items = [input_a, input_b].map { |i| options[:preserve_params_type] ? i : TensorStream.convert_to_tensor(i) }
      @data_type = set_data_type(options[:data_type])

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
        operands: hashify_tensor(items)
      }
    end

    def self.empty_matrix?(input)
      if input.is_a?(Array)
        input.each do |item|
          if item.is_a?(Array)
            return false unless empty_matrix?(item)
          elsif item != 0 || item != 0.0
            return false
          end
        end
      end

      true
    end

    def set_data_type(passed_data_type)
      case operation
      when :greater, :less, :equal, :not_equal, :greater_equal, :less_equal
        :boolean
      when :shape, :rank
        :int32
      when :random_normal, :random_uniform, :glorot_uniform
        passed_data_type || :float32
      else
        return passed_data_type if passed_data_type

        if @items[0]
          @items[0].data_type
        elsif @items[1]
          @items[1].data_type
        else
          :unknown
        end
      end
    end

    def to_math(name_only = false, max_depth = 99, _cur_depth = 0)
      return @name if max_depth.zero?

      sub_item = auto_math(items[0], name_only, max_depth - 1, _cur_depth + 1)
      sub_item2 = auto_math(items[1], name_only, max_depth - 1, _cur_depth + 1) if items[1]

      out = case operation
      when :argmax
        "argmax(#{sub_item},#{options[:axis]})"
      when :negate
        "-#{sub_item}"
      when :index
        "#{sub_item}[#{sub_item2}]"
      when :slice
        "#{sub_item}[#{sub_item2}]"
      when :assign_sub
        "(#{items[0] ? items[0].name : 'self'} -= #{auto_math(items[1], name_only, 1)})"
      when :assign_add
        "(#{items[0] ? items[0].name : 'self'} += #{auto_math(items[1], name_only, 1)})"
      when :assign
        "(#{items[0] ? items[0].name : 'self'} = #{auto_math(items[1], name_only, 1)})"
      when :sin, :cos, :tanh
        "#{operation}(#{sub_item})"
      when :add
        "(#{sub_item} + #{sub_item2})"
      when :sub
        "(#{sub_item} - #{sub_item2})"
      when :pow
        "(#{sub_item}^#{sub_item2})"
      when :div
        "(#{sub_item} / #{sub_item2})"
      when :mul
        if auto_math(items[0]) == 1
          sub_item2
        elsif auto_math(items[1]) == 1
          sub_item
        else
          "(#{sub_item} * #{sub_item2})"
        end
      when :sum
        "sum(|#{sub_item}|,  axis=#{sub_item2})"
      when :mean
        "mean(|#{sub_item}|, axis=#{sub_item2})"
      when :prod
        "prod(|#{sub_item}|,  axis=#{sub_item2})"
      when :gradients
        "gradient(#{sub_item})"
      when :stop_gradient
        sub_item
      when :matmul
        "#{sub_item}.matmul(#{sub_item2})"
      when :eye
        "eye(#{sub_item})"
      when :transpose
        "transpose(#{sub_item})"
      when :shape
        "#{sub_item}.shape"
      when :exp
        "e^#{sub_item})"
      when :ones
        "ones(#{sub_item})"
      when :ones_like
        "ones_like(#{sub_item})"
      when :flow_group
        "flow_group(#{items.collect { |i| auto_math(i, name_only, max_depth - 1, _cur_depth) }.join(',')})"
      when :zeros
        "zeros(#{sub_item})"
      when :reshape
        "reshape(#{sub_item},#{sub_item2})"
      when :rank
        "#{sub_item}.rank"
      when :cond
        "(#{auto_math(options[:pred], name_only, max_depth - 1, _cur_depth)} ? #{sub_item} : #{sub_item2})"
      when :less
        "#{sub_item} < #{sub_item2}"
      when :less_equal
        "#{sub_item} <= #{sub_item2}"
      when :greater
        "#{sub_item} > #{sub_item2}"
      when :greater_equal
        "#{sub_item} >= #{sub_item2}"
      when :square
        "#{sub_item}\u00B2"
      when :log
        "log(#{sub_item})"
      when :identity
        "identity(#{sub_item})"
      when :print
        "print(#{sub_item})"
      when :pad
        "pad(#{sub_item},#{auto_math(options[:paddings])})"
      when :equal
        "#{sub_item} == #{sub_item2}"
      when :not_equal
        "#{sub_item} != #{sub_item2}"
      when :logical_and
        "#{sub_item} && #{sub_item2}"
      when :sqrt
        "sqrt(#{sub_item})"
      when :log1p
        "log1p(#{sub_item})"
      when :zeros_like
        "zeros_like(#{sub_item})"
      when :where
        "where(#{auto_math(options[:pred], name_only, max_depth - 1, _cur_depth)}, #{sub_item}, #{sub_item2})"
      when :max
        "max(#{sub_item},#{sub_item2})"
      when :cast
        "cast(#{sub_item}, #{data_type})"
      when :broadcast_transform
        "broadcast_transform(#{sub_item},#{sub_item2})"
      when :broadcast_gradient_args
        "broadcast_transform(#{sub_item},#{sub_item2})"
      else
        "#{operation}(#{sub_item})" if sub_item
        "#{operation}(#{sub_item}, #{sub_item2})" if sub_item && sub_item2
      end
      ["\n",(_cur_depth + 1).times.collect { ' ' }, out].flatten.join
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
        item_shape = items[0].shape.shape
        return nil if item_shape.nil?
        return item_shape[1, item_shape.size]
      when :mean, :prod, :sum
        return [] if items[1].nil?
        return nil if items[0].nil?
        item_shape = items[0].shape.shape
        return nil if item_shape.nil?
        return nil if items[1].is_a?(Tensor) && items[1].value.nil?

        axis = items[1].is_a?(Tensor) ? items[1].value : items[1]

        axis = [ axis ] unless axis.is_a?(Array)
        return item_shape.each_with_index.map do |s, index|
          next nil if axis.include?(index)
          s
        end.compact
      when :reshape
        new_shape = items[1] && items[1].value ? items[1].value : nil
        return nil if new_shape.nil?

        item_shape = items[0].shape.shape
        return new_shape if item_shape.nil?

        return TensorShape.fix_inferred_elements(new_shape, item_shape.reduce(:*))
      when :flow_group
        return []
      when :zeros, :ones
        return items[0] ? items[0].value : options[:shape]
      when :zeros_like, :ones_like
        items[0].shape.shape
      when :shape
        return items[0].shape.shape ? [items[0].shape.shape.size] : nil
      when :matmul
        shape1 = items[0].shape.shape.nil? ? nil : items[0].shape.shape[0]
        shape2 = items[1].shape.shape.nil? ? nil : items[1].shape.shape[1]
        return [shape1, shape2]
      else
        return items[0].shape.shape if items.size == 1
        if items.size == 2 && items[0] && items[1]
          return TensorShape.infer_shape(items[0].shape.shape, items[1].shape.shape)
        end
      end

      nil
    end

    def propagate_consumer(consumer)
      super
      @items.compact.each do |item|
        item.send(:propagate_consumer, consumer) if item.name != name
      end
    end

    def propagate_outputs
      @items.compact.each do |item|
        item.send(:setup_output, self) if item.name != self.name
      end
    end

    def set_name
      "#{@operation}#{graph.get_operation_counter}:#{@rank}"
    end
  end
end
