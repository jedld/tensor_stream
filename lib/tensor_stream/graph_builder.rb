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
        # puts "build #{node['name']}"
        options = protobuf.options_evaluator(node)
        options[:name] = node['name']
        options[:__graph] = @graph
        value = options.delete('value')
        options = symbolize_keys(options)
        case node['op']
        when 'Const'
          dimension = shape_eval(value)
          rank = dimension.size
          options[:value] = value
          options[:const] = true
          TensorStream::Tensor.new(options[:dtype] || options[:T], rank, dimension, options)
        when 'VariableV2'
          # evaluate options
          shape = options[:shape]
          TensorStream::Variable.new(options[:dtype] || options[:T], nil, shape, nil, options)
        when 'Placeholder'
          shape = options[:shape]
          TensorStream::Placeholder.new(options[:dtype] || options[:T], nil, shape, options)
        else
          op = underscore(node['op']).to_sym
          unless TensorStream::Evaluator::RubyEvaluator.ops.keys.include?(op)
            puts "warning unsupported op #{op}"
          end
          # map input tensor
          inputs = node['input'].map do |input|
            input[0] = '' if input.start_with?('^')

            input_indexed, index = input.split(':')

            tensor = if index && index.to_i > 0
                       @graph.get_tensor_by_name(input_indexed)[index.to_i]
                     else
                       @graph.get_tensor_by_name(input)
                     end

            raise "tensor not found by name #{input}" if tensor.nil?

            tensor
          end

          options[:data_type] = options.delete(:T)
          TensorStream::Operation.new(op, *inputs, options)
        end
      end

      @graph
    end
  end
end