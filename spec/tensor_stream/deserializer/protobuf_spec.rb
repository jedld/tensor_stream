require 'spec_helper'

RSpec.describe TensorStream::Protobuf do
  context ".load" do
    specify "Parses a pb file and turns it into a ruby hash" do
      protobuf = TensorStream::Protobuf.new
      parsed_tree = protobuf.load(File.join('spec','fixtures','tensorflow.proto'))

      expect(parsed_tree). to eq([{"type"=>"node",
        "name"=>"Const",
        "op"=>"Const",
        "attributes"=>
         [{"key"=>"dtype", "value"=>{"type"=>"DT_FLOAT"}},
          {"key"=>"value", "value"=>{"tensor"=>{"dtype"=>"DT_FLOAT", "tensor_content"=>"\\000\\000\\200?\\315\\314\\214?\\315\\314\\014@33S@\\315\\314\\214?\\315\\314\\014@33S@\\000\\000\\200@"}}}]},
       {"type"=>"node", "name"=>"Const_1", "op"=>"Const", "attributes"=>[{"key"=>"dtype", "value"=>{"type"=>"DT_INT32"}}, {"key"=>"value", "value"=>{"tensor"=>{"dtype"=>"DT_INT32", "int_val"=>"2"}}}]},
       {"type"=>"node", "name"=>"v1/initial_value", "op"=>"Const", "attributes"=>[{"key"=>"dtype", "value"=>{"type"=>"DT_FLOAT"}}, {"key"=>"value", "value"=>{"tensor"=>{"dtype"=>"DT_FLOAT", "float_val"=>"1.0"}}}]},
       {"type"=>"node",
        "name"=>"v1",
        "op"=>"VariableV2",
        "attributes"=>[{"key"=>"container", "value"=>{"type"=>"\"\""}}, {"key"=>"dtype", "value"=>{"type"=>"DT_FLOAT"}}, {"key"=>"shape", "value"=>{}}, {"key"=>"shared_name", "value"=>{"type"=>"\"\""}}]},
       {"type"=>"node",
        "name"=>"v1/Assign",
        "op"=>"Assign",
        "input"=>["v1", "v1/initial_value"],
        "attributes"=>[{"key"=>"T", "value"=>{"type"=>"DT_FLOAT"}}, {"key"=>"_class", "value"=>{}}, {"key"=>"use_locking", "value"=>{"type"=>"true"}}, {"key"=>"validate_shape", "value"=>{"type"=>"true"}}]},
       {"type"=>"node", "name"=>"v1/read", "op"=>"Identity", "input"=>["v1"], "attributes"=>[{"key"=>"T", "value"=>{"type"=>"DT_FLOAT"}}, {"key"=>"_class", "value"=>{}}]},
       {"type"=>"node", "name"=>"Const_2", "op"=>"Const", "attributes"=>[{"key"=>"dtype", "value"=>{"type"=>"DT_FLOAT"}}, {"key"=>"value", "value"=>{"tensor"=>{"dtype"=>"DT_FLOAT", "float_val"=>"2.0"}}}]},
       {"type"=>"node", "name"=>"mul_1", "op"=>"Mul", "input"=>["Const", "Const_2"], "attributes"=>[{"key"=>"T", "value"=>{"type"=>"DT_FLOAT"}}]},
       {"type"=>"node", "name"=>"add", "op"=>"Add", "input"=>["mul_1", "v1/read"], "attributes"=>[{"key"=>"T", "value"=>{"type"=>"DT_FLOAT"}}]},
       {"type"=>"versions", "producer"=>"26"}])
    end
  end
end