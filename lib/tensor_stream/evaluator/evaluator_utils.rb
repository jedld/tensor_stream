module TensorStream
  class EvaluatorUtils
    extend TensorStream::StringHelper

    def self.get_evaluator_classes(evaluators)
      @evaluator_classes ||= if evaluators.is_a?(Array)
        if evaluators.empty?
          TensorStream::Evaluator.default_evaluators
        else
          evaluators.collect { |name| Object.const_get("TensorStream::Evaluator::#{camelize(name.to_s)}") }
        end
      elsif evaluators.nil?
        TensorStream::Evaluator.default_evaluators
      else
        [Object.const_get("TensorStream::Evaluator::#{camelize(evaluators.to_s)}")]
      end
      @evaluator_classes
    end
  end
end