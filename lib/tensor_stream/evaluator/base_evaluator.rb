module TensorStream
  module Evaluator
    class BaseEvaluator
      def self.query_supported_devices
        [Device.new("cpu", :cpu, "ruby")]
      end
    end

    def self.evaluators
      @evaluators ||= {}
    end

    def self.register_evaluator(klass, name)
      @evaluators ||= {}
      @evaluators[name] = { name: name, class: klass }
    end
  end
end
