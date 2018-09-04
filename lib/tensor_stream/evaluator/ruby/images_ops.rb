module TensorStream
  module ImagesOps
    def ImagesOps.included(klass)
      klass.class_eval do
        register_op :decode_png do |_context, _tensor, inputs|

        end
      end
    end
  end
end