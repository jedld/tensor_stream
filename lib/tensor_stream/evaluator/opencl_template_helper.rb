require 'erb'
class OpenclTemplateHelper
  def initialize(source)
    @source = source
  end

  def generate
    ERB.new(@source, nil, '%').result(binding)
  end

  def render(template, locals = {})
    filename = File.join(File.dirname(__FILE__), 'kernels', "_#{template}")
    source = File.read(filename)
    current_scope = binding
    locals.each do |k,v|
      current_scope.local_variable_set(k.to_sym, v)
    end
    ERB.new(source, nil, '%').result(current_scope)
  end

  def dtype_to_c_type(dtype)
    case(dtype)
    when 'fp'
      'float'
    when 'int'
      'int'
    end
  end

  def operator_to_c(op)
    case(op)
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
    else
      raise "unsupported op #{op}"
    end
  end
end