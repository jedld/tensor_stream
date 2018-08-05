module TensorStream
  class GraphBuilder
    include TensorStream::OpHelper
    include TensorStream::StringHelper

    attr_accessor :graph

    def initialize(graph)
      @graph = graph
    end

    def build(buffer)
      protobuf = TensorStream::Protobuf.new
      parsed_tree = protobuf.load_from_string(buffer)
      parsed_tree.each do |node|
        next unless node['type'] == 'node'

        options = protobuf.options_evaluator(node)
        options[:name] = node['name']
        options[:__graph] = @graph
        value = options.delete('value')
        options = symbolize_keys(options)
        case node['op']
        when 'Const'
          shape = shape_eval(value)
          options[:value] = value
          TensorStream::Tensor.new(options[:dtype], shape.size, shape, options)
        when 'VariableV2'
          # evaluate options
          shape = options[:shape]
          TensorStream::Variable.new(options[:dtype], nil, shape, nil, options)
        when 'Placeholder'
          shape = options[:shape]
          TensorStream::Placeholder.new(options[:dtype], nil, shape, options)
        else
          op = underscore(node['op']).to_sym
          unless TensorStream::Evaluator::RubyEvaluator.ops.keys.include?(op)
            puts "warning unsupported op #{op}"
            binding.pry
          end
          # map input tensor
          inputs = node['input'].map do |input|
            input[0] = '' if input.start_with?('^')

            tensor = @graph.get_tensor_by_name(input)
            raise "tensor not found by name #{input}" if tensor.nil?

            tensor
          end
          TensorStream::Operation.new(op, *inputs, options)
        end
      end

      @graph
    end
  end
end