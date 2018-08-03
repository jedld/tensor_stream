module TensorStream
  module Train
    module Utils
      def create_global_step(graph = nil)
        target_graph = graph || TensorStream.get_default_graph
        raise TensorStream::ValueError, '"global_step" already exists.' unless get_global_step(target_graph).nil?

        TensorStream.variable_scope.get_variable(
          TensorStream::GraphKeys::GLOBAL_STEP, shape: [],
                                                dtype: :int64,
                                                initializer: TensorStream.zeros_initializer,
                                                trainable: false,
                                                collections: [TensorStream::GraphKeys::GLOBAL_VARIABLES,
                                                              TensorStream::GraphKeys::GLOBAL_STEP])

      end

      def get_global_step(graph = nil)
        target_graph = graph || TensorStream.get_default_graph
        global_step_tensors = target_graph.get_collection(TensorStream::GraphKeys::GLOBAL_STEP)
        global_step_tensor = if global_step_tensors.nil? || global_step_tensors.empty?
                               target_graph.get_tensor_by_name('global_step:0')
                             elsif global_step_tensors.size == 1
                               global_step_tensors[0]
                             else
                               TensorStream.logger.error("Multiple tensors in global_step collection.")
                               nil
                             end
        global_step_tensor
      end
    end
  end
end