##
# Ruby port of tensorflow's learning rate decay functions
module TensorStream
  module Train
    module LearningRateDecay
      include TensorStream::Utils
      include TensorStream::OpHelper
      include TensorStream::Ops

      ##
      # Applies exponential decay to the learning rate
      def exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase: false, name: nil)
        raise TensorStream::ValueError, "global_step is required for exponential_decay." if global_step.nil?

        name_scope(name, default: "ExponentialDecay", values: [learning_rate, global_step, decay_steps, decay_rate]) do
          learning_rate = convert_to_tensor(learning_rate, name: "learning_rate")
          data_type = learning_rate.data_type
          decay_steps = cast(decay_steps, data_type)
          decay_rate = cast(decay_rate, data_type)

          global_step_recomp = cast(global_step, data_type)
          p = global_step_recomp / decay_steps
          p = floor(p) if staircase
          multiply(learning_rate, pow(decay_rate, p), name: name)
        end
      end
    end
  end
end
