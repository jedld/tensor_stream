module TensorStream
  module TensorUtils

    # Returns the constant value of the given tensor, if efficiently calculable.
    # This function attempts to partially evaluate the given tensor, and
    # returns its value as a numpy ndarray if this succeeds.
    def constant_value(tensor, partial: false)
      ret = _constvalue(tensor, partial)
      tensor.graph.prevent_feeding(tensor) unless ret.nil?
      ret
    end

    def _constvalue(tensor, partial)
      TensorStream::OpMaker.constant_op(tensor, partial)
    end
  end
end