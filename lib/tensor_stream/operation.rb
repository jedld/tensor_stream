module TensorStream
  class Operation < Tensor
    attr_accessor :name, :operation, :items, :rank, :options

    def initialize(operation, a, b, options = {})
      @operation = operation
      @rank = options[:rank] || 0
      @name = options[:name] || set_name
      @internal = options[:internal]
      @given_name = @name
      @source = set_source(caller_locations)

      @graph = options[:graph] || TensorStream.get_default_graph
      @options = options


      @items = [a, b].map { |i| options[:preserve_params_type] ? i : auto_wrap(i) }
      @data_type = set_data_type(options[:data_type])

      if options[:shape]
        @shape = TensorShape.new(options[:shape], options[:shape].size || 0)
      end
      @graph.add_node(self)
    end
    def to_s
      @name
    end

    def self.reset_counters
      @@op_counter = 0
    end

    def to_h
      {
        op: operation,
        name: name,
        operands: hashify_tensor(items)
      }
    end

    def self.empty_matrix?(m)
      if m.kind_of?(Array)
        m.each do |item|
          if item.kind_of?(Array)
            return false if !empty_matrix?(item)
          else
            return false if item!=0 || item!=0.0
          end
        end
      end

      return true
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
      return @name if max_depth == 0

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
        "(#{items[0] ? items[0].name : "self"} -= #{auto_math(items[1], name_only)})"
      when :assign_add
        "(#{items[0] ? items[0].name : "self"} += #{auto_math(items[1], name_only)})"
      when :assign
        "(#{items[0] ? items[0].name : "self"} = #{auto_math(items[1], name_only)})"
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
        "flow_group(#{items.collect { |i| auto_math(i)}.join(',')})"
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
      when :sqrt
        "sqrt(#{sub_item})"
      when :zeros_like
        "zeros_like(#{sub_item})"
      when :where
        "where(#{auto_math(options[:pred] , name_only, max_depth - 1)},#{auto_math(items[0])},#{auto_math(items[1])})"
      when :max
        "max(#{auto_math(sub_item)},#{auto_math(items[1])})"
      when :cast
        "cast(#{auto_math(sub_item)}, #{data_type})"
      else
        fail "math form for #{operation}"
      end
    end

    def run
      self.eval
    end

    private

    def self.operation_counter
      @@op_counter ||= 0

      name = if @@op_counter == 0
        ""
      else
        "_#{@@op_counter}"
      end

      @@op_counter += 1
      
      name
    end

    def set_name
      "#{@operation}#{Operation.operation_counter}:#{@rank}"
    end
  end
end