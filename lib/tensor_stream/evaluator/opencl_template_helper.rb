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
end