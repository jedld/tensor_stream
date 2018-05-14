module TensorStream
  # TensorStream class that defines an operation
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options
    attr_reader :outputs

    def initialize(operation, input_a, input_b, options = {})
      @graph = options[:graph] || TensorStream.get_default_graph

      @operation = operation
      @rank = options[:rank] || 0
      @name = options[:name] || set_name
      @internal = options[:internal]
      @given_name = @name
      @source = format_source(caller_locations)

      @options = options

      @items = [input_a, input_b].map { |i| options[:preserve_params_type] ? i : auto_wrap(i) }
      @data_type = set_data_type(options[:data_type])

      @shape = TensorShape.new(options[:shape], options[:shape].size || 0) if options[:shape]

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
      when :greater, :less, :equal
        :boolean
      when :shape, :rank
        :int32
      else
        passed_data_type || (@items[0] ? @items[0].data_type : :unknown)
      end
    end

    def to_math(name_only = false, max_depth = 99)
      return @name if max_depth.zero?

      sub_item = auto_math(items[0], name_only, max_depth - 1)

      case operation
      when :argmax
        "argmax(#{auto_math(items[0])},#{options[:axis]})"
      when :negate
        "-#{sub_item}"
      when :index
        "#{sub_item}[#{auto_math(items[1], name_only, max_depth - 1)}]"
      when :slice
        "#{sub_item}[#{auto_math(items[1], name_only, max_depth - 1)}]"
      when :assign_sub
        "(#{items[0] ? items[0].name : 'self'} -= #{auto_math(items[1], name_only)})"
      when :assign_add
        "(#{items[0] ? items[0].name : 'self'} += #{auto_math(items[1], name_only)})"
      when :assign
        "(#{items[0] ? items[0].name : 'self'} = #{auto_math(items[1], name_only)})"
      when :sin, :cos, :tanh
        "#{operation}(#{sub_item})"
      when :add
        "(#{sub_item} + #{auto_math(items[1], name_only, max_depth - 1)})"
      when :sub
        "(#{sub_item} - #{auto_math(items[1], name_only, max_depth - 1)})"
      when :pow
        "(#{sub_item}^#{auto_math(items[1], name_only, max_depth - 1)})"
      when :div
        "(#{sub_item} / #{auto_math(items[1], name_only, max_depth - 1)})"
      when :mul
        if auto_math(items[0]) == 1
          auto_math(items[1], name_only, max_depth - 1)
        elsif auto_math(items[1]) == 1
          sub_item
        else
          "(#{sub_item} * #{auto_math(items[1], name_only, max_depth - 1)})"
        end
      when :reduce_sum
        "reduce_sum(|#{sub_item}|)"
      when :reduce_mean
        "reduce_mean(|#{sub_item}|)"
      when :reduce_prod
        "reduce_prod(|#{sub_item}|)"
      when :gradients
        "gradient(#{sub_item})"
      when :stop_gradient
        sub_item
      when :matmul
        "#{sub_item}.matmul(#{auto_math(items[1], name_only, max_depth - 1)})"
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
        "flow_group(#{items.collect { |i| auto_math(i) }.join(',')})"
      when :zeros
        "zeros(#{sub_item})"
      when :reshape
        "reshape(#{sub_item},#{auto_math(items[1], name_only, max_depth - 1)})"
      when :rank
        "#{sub_item}.rank"
      when :cond
        "(#{auto_math(options[:pred])} ? #{sub_item} : #{auto_math(items[1], name_only, max_depth - 1)})"
      when :less
        "#{sub_item} < #{auto_math(items[1], name_only, max_depth - 1)}"
      when :greater
        "#{sub_item} > #{auto_math(items[1], name_only, max_depth - 1)}"
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
        "#{sub_item} == #{auto_math(items[1], name_only, max_depth - 1)}"
      when :not_equal
        "#{sub_item} != #{auto_math(items[1], name_only, max_depth - 1)}"
      when :logical_and
        "#{sub_item} && #{auto_math(items[1], name_only, max_depth - 1)}"
      when :sqrt
        "sqrt(#{sub_item})"
      when :zeros_like
        "zeros_like(#{sub_item})"
      when :where
        "where(#{auto_math(options[:pred], name_only, max_depth - 1)},#{auto_math(items[0])},#{auto_math(items[1])})"
      when :max
        "max(#{auto_math(sub_item)},#{auto_math(items[1])})"
      when :cast
        "cast(#{auto_math(sub_item)}, #{data_type})"
      else
        raise "no math form for #{operation} defined"
      end
    end

    def run
      eval
    end

    private

    def propagate_consumer(consumer)
      super(consumer)

      @items.compact.each do |item|
        binding.pry unless item.is_a?(Tensor)
        item.send(:propagate_consumer, consumer) if item.name!=self.name
      end
    end

    def set_name
      "#{@operation}#{graph.get_operation_counter}:#{@rank}"
    end
  end
end
