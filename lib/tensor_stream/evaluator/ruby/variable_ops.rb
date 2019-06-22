module TensorStream
  ## Collection of machine learning related ops
  module VariableOps
    def self.included(klass)
      klass.class_eval do
        register_op :variable_v2 do |_context, tensor, _inputs|
          value = var_read_value(tensor)
          raise "variable #{tensor.options[:var_name]} not initalized" if value.nil?

          value
        end

        register_op :assign do |context, tensor, inputs|
          var_assign_value(tensor, inputs[0])
        end

        register_op :assign_add, no_eval: true do |context, tensor, inputs|
          current_val = var_read_value(tensor)

          raise "variable #{tensor.options[:var_name]} not initialized" if current_val.nil?
          eval_a, eval_b = broadcast(current_val, inputs[0])
          result = multi_array_op(->(var, val) { var + val }, eval_a, eval_b)
          var_assign_value(tensor, result)
        end

        register_op :assign_sub do |context, tensor, _inputs|
          current_val = var_read_value(tensor)
          raise "variable #{tensor.options[:var_name]} not initialized" if current_val.nil?
          eval_a, eval_b = broadcast(current_val, inputs[0])
          result = multi_array_op(->(var, val) { var - val }, eval_a, eval_b)
          var_assign_value(tensor, result)
        end

        register_op :save_ts do |_context, tensor, inputs|
          outputfile = inputs[0]
          inputs = tensor.inputs.dup

          inputs.shift
          variables = {}
          inputs.each do |savable|
            val = var_read_value(savable)

            packed_data = Zlib::Deflate.deflate(TensorStream::Packer.pack(val, savable.data_type))
            variables[savable.options[:var_name]] = {
              "shape" => shape_eval(val),
              "data" => Base64.strict_encode64(packed_data),
            }
          end

          File.write(outputfile, {"variables" => variables}.to_yaml)
          nil
        end

        register_op :restore_ts do |_context, tensor, inputs|
          inputs = inputs.dup
          filename = inputs.shift
          tensor_names = inputs

          input_dump = YAML.safe_load(File.read(filename), [Symbol])
          vars = tensor.graph.get_collection(GraphKeys::GLOBAL_VARIABLES)
          vars.select! { |v| input_dump["variables"].key?(v.name) && tensor_names.include?(v.name) }
          vars.each do |variable|
            data = TensorStream::Packer.unpack(Zlib::Inflate.inflate(Base64.decode64(input_dump["variables"][variable.name]["data"])), variable.data_type)
            shape = input_dump["variables"][variable.name]["shape"]
            variable.buffer = nil
            var_assign_value(variable, TensorShape.reshape(data, shape))
          end

          nil
        end
      end
    end
  end
end