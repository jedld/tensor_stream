class TensorStream::OpMaker
  attr_reader :operation, :description, :parameters,
              :options, :gradient, :check_types,
              :supports_broadcast, :data_type_coercion,
              :aliases, :custom, :infer_type_proc, :exclude,
              :data_type_block

  def initialize(op)
    @operation = op
    @parameters = []
    @options = {}
    @gradient = nil
    @supports_broadcast = false
    @data_type_coercion = false
    @exclude = false
    @description = []
    @aliases = []
    @custom = []
    @infer_type_proc = lambda { |tensor|
      next nil if tensor.inputs[0].nil?
      next tensor.inputs[0].shape.shape if tensor.inputs.size == 1

      TensorStream::TensorShape.infer_shape(tensor.inputs[0].shape.shape, tensor.inputs[1].shape.shape) if tensor.inputs.size == 2 && tensor.inputs[0] && tensor.inputs[1]
    }
  end

  def other_names(aliases)
    @aliases += aliases
  end

  def add_custom(custom_code)
    @custom << custom_code
  end

  def self.scan
    op_files = Dir[File.join(File.dirname(__FILE__), "ops", "*.rb")]
    op_files.each { |file|
      load File.join("tensor_stream", "ops", File.basename(file))
    }
  end

  def self.define_operation(op_code, &block)
    @ops ||= {}
    op_maker = TensorStream::OpMaker.new(op_code.to_sym)
    block.call(op_maker)
    @ops[op_code.to_sym] = op_maker
  end

  # call an operations' gradient definition
  def self.gradient_op(context_caller, node, grad)
    raise "No derivative op defined for #{node.operation}" if @ops[node.operation].nil? || @ops[node.operation].gradient.nil?

    context_caller.instance_exec(grad, node, node.inputs, &@ops[node.operation].gradient)
  end

  def self.infer_shape(context_caller, tensor)
    return nil unless @ops[tensor.operation]

    context_caller.instance_exec(tensor, &@ops[tensor.operation].infer_type_proc)
  end

  def self.infer_data_type(context_caller, tensor, passed_data_type)
    return passed_data_type if passed_data_type

    if @ops[tensor.operation] && @ops[tensor.operation].data_type_block
      context_caller.instance_exec(tensor, &@ops[tensor.operation].data_type_block)
    else
      if tensor.inputs[0]
        tensor.inputs[0].data_type
      elsif tensor.inputs[1]
        tensor.inputs[1].data_type
      else
        :unknown
      end
    end
  end

  def self.each_op(&block)
    @ops.values.sort_by { |op| op.operation }.reject(&:exclude).each do |op|
      block.call(op)
    end
  end

  def what_it_does(description)
    @description << description
  end

  def what_it_does_code(description)
    @description << "<tt>#{description}</tt>"
  end

  def exclude!
    @exclude = true
  end

  def description_lines
    description.map { |line| line.split("\n") }.flatten
  end

  def generate_body
    body = []
    parameters.select { |p| p[:validate] }.each do |p|
      body << "check_allowed_types(#{p[:name]}, TensorStream::Ops::#{p[:validate]})"
    end
    if data_type_coercion?
      body << "#{expand_params(false).join(', ')} = apply_data_type_coercion(#{expand_params(false).join(', ')})"
    end
    if check_types?
      body << "check_data_types(#{expand_params(false).join(', ')})"
    end
    custom.each do |c|
      body << c
    end
    body << "_op(:#{operation}, #{(expand_params(false) + options_call).join(', ')})"
    body.map { |line| "      #{line}"}.join("\n")
  end

  ##
  # adds a parameter to the op
  #
  def parameter(name, description, default_value = nil, validate: nil)
    @parameters << {
      name: name.to_s,
      description: description,
      default_value: default_value,
      validate: validate
    }
  end

  def option(name, description, default_value = nil, options = {})
    @options[name] = { description: description, default_value: default_value, options: options }
  end

  def define_gradient(&block)
    @gradient = block
  end

  def define_shape(&block)
    @infer_type_proc = block
  end

  def define_data_type(&block)
    @data_type_block = block
  end

  def expand_params(print_defaults)
    @parameters.map { |param|
      print_defaults && param[:default_value] ? "#{param[:name]} = #{default_with_nil(param[:default_value])}" : "#{param[:name]}"
    }
  end

  def parameters_must_have_same_data_type!
    @check_types = true
  end

  def apply_data_type_coercion!
    @data_type_coercion = true
  end

  def supports_broadcasting!
    if (@parameters.size> 1)
      @supports_broadcast = true
    else
      raise "Ops with parameters < 2 cannot support broadcasting"
    end
  end

  def supports_broadcasting?
    @supports_broadcast
  end

  def data_type_coercion?
    @data_type_coercion
  end

  def check_types?
    @check_types
  end

  def expand_options(print_defaults)
    @options.map { |k, v|
      print_defaults && v[:default_value] ? "#{k}: #{default_with_nil(v[:default_value])}" : "#{k}:"
    }
  end

  def options_call
    @options.reject { |k, v| v.dig(:options, :exclude) }.map { |k, v|
      if v.dig(:options, :alias)
        "#{v.dig(:options, :alias)}: #{k}"
      else
        "#{k}: #{k}"
      end
    }
  end

  def default_with_nil(v)
    v == :nil ? 'nil' : v
  end
end

TensorStream::OpMaker.scan
