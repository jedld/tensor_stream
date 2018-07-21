require 'erb'
class OpenclTemplateHelper
  def initialize(source)
    @source = source
  end

  def generate(args = {})
    current_scope = binding

    args.each do |k, v|
      current_scope.local_variable_set(k.to_sym, v)
    end

    ERB.new(@source, nil, '%').result(current_scope)
  end

  def floating_point?(dtype)
    TensorStream::Ops::FLOATING_POINT_TYPES.include?(dtype)
  end

  def render(template, locals = {})
    filename = File.join(File.dirname(__FILE__), 'kernels', "_#{template}")
    source = File.read(filename)
    current_scope = binding
    locals.each do |k, v|
      current_scope.local_variable_set(k.to_sym, v)
    end
    ERB.new(source, nil, '%').result(current_scope)
  end

  def dtype_to_c_type(dtype)
    case dtype.to_s
    when 'float64'
      'double'
    when 'float32', 'float'
      'float'
    when 'int32', 'int'
      'int'
    when 'int16'
      'short'
    when 'boolean'
      'short'
    else
      raise "unknown dtype #{dtype}"
    end
  end

  def min_value_for(dtype)
    case dtype.to_s
    when 'float64'
      'DBL_MIN'
    when 'float32', 'float'
      'FLT_MIN'
    when 'int32', 'int'
      'INT_MIN'
    when 'int16'
      'SHRT_MIN'
    when 'boolean'
      '0'
    else
      raise "unknown dtype #{dtype}"
    end
  end

  def operator_to_c(op)
    case op
    when 'less'
      '<'
    when 'less_equal'
      '<='
    when 'equal'
      '=='
    when 'greater'
      '>'
    when 'greater_equal'
      '>='
    when 'not_equal'
      '!='
    when 'logical_and'
      '&&'
    when 'div'
      '/'
    when 'add'
      '+'
    when 'sub'
      '-'
    when 'mul'
      '*'
    when 'mod'
      '%'
    else
      raise "unsupported op #{op}"
    end
  end
end