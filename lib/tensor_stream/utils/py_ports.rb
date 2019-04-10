module TensorStream
  ##
  # Utility functions that emulate python operators
  module PyPorts
    def floor_div(a, b)
      if a.is_a?(Float)
        (a.to_i / b.to_i).to_f
      else
        a / b
      end
    end
  end
end