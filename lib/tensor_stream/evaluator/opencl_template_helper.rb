require 'erb'
class OpenclTemplateHelper
  def initialize(source)
    @source = source
  end

  def generate
    ERB.new(@source, nil, '%').result(binding)
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
    else
      raise "unsupported op #{op}"
    end
  end
end