module TensorStream
  module MathOps
    def MathOps.included(klass)
      klass.class_eval do
        register_op :tanh, no_eval: true do |context, _tensor, inputs|
          call_op(:tanh, inputs[0], context, ->(t, _b) { Math.tanh(t) })
        end

        register_op :tan, no_eval: true do |context, _tensor, inputs|
          call_op(:tan, inputs[0], context, ->(t, _b) { Math.tan(t) })
        end

        register_op :atan, no_eval: true do |context, _tensor, inputs|
          call_op(:atan, inputs[0], context, ->(t, _b) { Math.atan(t) })
        end

        register_op :sec, no_eval: true do |context, _tensor, inputs|
          call_op(:sec, inputs[0], context, ->(t, _b) { Math.sec(t) })
        end

        register_op :sin, no_eval: true do |context, _tensor, inputs|
          call_op(:sin, inputs[0], context, ->(t, _b) { Math.sin(t) })
        end

        register_op :add, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :add, a, b, context, ->(t, u) { t + u })
        end

        register_op :add_n, no_eval: true do |context, tensor, inputs|
          if inputs.size == 1
            complete_eval(inputs[0], context)
          elsif inputs.size > 1

            a = inputs.pop
            until inputs.empty?
              b = inputs.pop
              a = call_vector_op(tensor, :add, a, b, context, ->(t, u) { t + u })
            end
            a
          end
        end

        register_op :sub, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :sub, a, b, context, ->(t, u) { t - u })
        end

        register_op %i[floor_mod mod], no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :mod, a, b, context, ->(t, u) { t % u })
        end

        register_op %i[floor_div], no_eval: true do |context, tensor, inputs|
          a, b = inputs
          if fp_type?(tensor.data_type)
            call_vector_op(tensor, :div, a, b, context, ->(t, u) { (t / u).to_i.to_f })
          else
            call_vector_op(tensor, :div, a, b, context, ->(t, u) { t / u })
          end
        end

        register_op :mul, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :mul, a, b, context, ->(t, u) { t * u })
        end

        register_op :pow, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :pow, a, b, context, ->(t, u) { t**u })
        end

        register_op :squared_difference, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :squared_difference, a, b, context, ->(t, u) { (t - u) * (t - u) })
        end

        register_op :round, no_eval: true do |context, _tensor, inputs|
          call_op(:round, inputs[0], context, ->(t, _b) { t.round })
        end

        register_op :abs, no_eval: true do |context, _tensor, inputs|
          call_op(:abs, inputs[0], context, ->(t, _b) { t.abs })
        end

        register_op :asin, no_eval: true do |context, _tensor, inputs|
          call_op(:asin, inputs[0], context, ->(t, _b) { Math.asin(t) })
        end

        register_op :acos, no_eval: true do |context, _tensor, inputs|
          call_op(:acos, inputs[0], context, ->(t, _b) { Math.acos(t) })
        end

        register_op :cos, no_eval: true do |context, _tensor, inputs|
          call_op(:cos, inputs[0], context, ->(t, _b) { Math.cos(t) })
        end

        register_op :log1p, no_eval: true do |context, _tensor, inputs|
          call_op(:log1p, inputs[0], context, ->(t, _b) { Math.log(1 + t) })
        end

        register_op :log, no_eval: true do |context, _tensor, inputs|
          call_op(:log, inputs[0], context, ->(t, _b) { t < 0 ? Float::NAN : Math.log(t) })
        end

        register_op :exp, no_eval: true do |context, _tensor, inputs|
          call_op(:exp, inputs[0], context, ->(t, _b) { Math.exp(t) })
        end

        register_op :sigmoid, no_eval: true do |context, _tensor, inputs|
          call_op(:sigmoid, inputs[0], context, ->(t, _b) { sigmoid(t) })
        end

        register_op :sqrt, no_eval: true do |context, _tensor, inputs|
          call_op(:sqrt, inputs[0], context, ->(t, _b) { Math.sqrt(t) })
        end

        register_op :floor, no_eval: true do |context, _tensor, inputs|
          call_op(:floor, inputs[0], context, ->(t, _b) { t.floor })
        end

        register_op :ceil, no_eval: true do |context, _tensor, inputs|
          call_op(:ceil, inputs[0], context, ->(t, _b) { t.ceil })
        end

        register_op :square, no_eval: true do |context, _tensor, inputs|
          call_op(:square, inputs[0], context, ->(t, _b) { t * t })
        end

        register_op :reciprocal, no_eval: true do |context, _tensor, inputs|
          call_op(:reciprocal, inputs[0], context, ->(t, _b) { 1 / t })
        end

        register_op %i[neg negate], no_eval: true do |context, tensor, inputs|
          call_vector_op(tensor, :negate, inputs[0], nil, context, ->(t, _u) { -t })
        end

        register_op :tanh_grad, no_eval: true do |context, _tensor, inputs|
          call_op(:tanh_grad, inputs[0], context, ->(t, _b) { 1 - Math.tanh(t) * Math.tanh(t) })
        end
      end
    end
  end
end