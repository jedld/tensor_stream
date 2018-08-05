require 'spec_helper'

RSpec.describe TensorStream::Protobuf do
  let(:instance) { TensorStream::Protobuf.new }

  context ".load" do
    specify "Parses a pb file and turns it into a ruby hash" do
      parsed_tree = instance.load(File.join('spec','fixtures','tensorflow.proto'))

      expect(parsed_tree). to eq([
        {
           "type"       => "node",
           "name"       => "Const",
           "op"         => "Const",
           "attributes" => [
              {
                 "key"   => "dtype",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "value",
                 "value" => {
                    "tensor" => {
                       "dtype"          => "DT_FLOAT",
                       "shape"          => [
                          2,
                          4
                       ],
                       "tensor_content" => "\\000\\000\\200?\\315\\314\\214?\\315\\314\\014@33S@\\315\\314\\214?\\315\\314\\014@33S@\\000\\000\\200@"
                    }
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "Const_1",
           "op"         => "Const",
           "attributes" => [
              {
                 "key"   => "dtype",
                 "value" => {
                    "type" => "DT_INT32"
                 }
              },
              {
                 "key"   => "value",
                 "value" => {
                    "tensor" => {
                       "dtype"   => "DT_INT32",
                       "shape"   => [],
                       "int_val" => "2"
                    }
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "v1/initial_value",
           "op"         => "Const",
           "attributes" => [
              {
                 "key"   => "dtype",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "value",
                 "value" => {
                    "tensor" => {
                       "dtype"     => "DT_FLOAT",
                       "shape"     => [],
                       "float_val" => "1.0"
                    }
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "v1",
           "op"         => "VariableV2",
           "attributes" => [
              {
                 "key"   => "container",
                 "value" => {
                    "s" => ""
                 }
              },
              {
                 "key"   => "dtype",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "shape",
                 "value" => {"shape"=>[]}
              },
              {
                 "key"   => "shared_name",
                 "value" => {
                    "s" => ""
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "v1/Assign",
           "op"         => "Assign",
           "input"      => [
              "v1",
              "v1/initial_value"
           ],
           "attributes" => [
              {
                 "key"   => "T",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "_class",
                 "value"=>[{"s"=>" \"loc"}]
              },
              {
                 "key"   => "use_locking",
                 "value" => {
                    "b" => "true"
                 }
              },
              {
                 "key"   => "validate_shape",
                 "value" => {
                    "b" => "true"
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "v1/read",
           "op"         => "Identity",
           "input"      => [
              "v1"
           ],
           "attributes" => [
              {
                 "key"   => "T",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "_class",
                 "value"=>[{"s"=>" \"loc"}]
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "Const_2",
           "op"         => "Const",
           "attributes" => [
              {
                 "key"   => "dtype",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              },
              {
                 "key"   => "value",
                 "value" => {
                    "tensor" => {
                       "dtype"     => "DT_FLOAT",
                       "shape"     => [],
                       "float_val" => "2.0"
                    }
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "mul_1",
           "op"         => "Mul",
           "input"      => [
              "Const",
              "Const_2"
           ],
           "attributes" => [
              {
                 "key"   => "T",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              }
           ]
        },
        {
           "type"       => "node",
           "name"       => "add",
           "op"         => "Add",
           "input"      => [
              "mul_1",
              "v1/read"
           ],
           "attributes" => [
              {
                 "key"   => "T",
                 "value" => {
                    "type" => "DT_FLOAT"
                 }
              }
           ]
        },
        {
           "type"     => "versions",
           "producer" => "26"
        }
     ])
    end
  end

  context ".parse_value" do
    let(:float_arr) {[[2.0, 3.0, 4.0, 1.1], [1.1, 2.2, 1.0, 4.0]]}
    let(:int_arr) {[[1, 2, 4, 1], [1, 2, 3, 25]]}

    def pack_arr_float(float_arr)
      float_arr.flatten.pack('f*').bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr  }.join
    end

    def pack_arr_int(int_arr)
      int_arr.flatten.pack('l*').bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr  }.join
    end

    def get_value(value, shape, type)
      {
        "tensor" => {
           "dtype"          => type,
           "shape"          => shape,
           "tensor_content" => type == 'DT_FLOAT' ? pack_arr_float(value) : pack_arr_int(value)
        }
      }
    end

    def get_scalar_value(value, type)
      {
        "tensor" => {
           "dtype"          => type,
           "shape"          => [],
           (type == 'DT_FLOAT' ? "float_val" : "int_val") => value
        }
      }
    end

    specify "converts tensor value node to ruby" do
      expect(tr(instance.parse_value(get_value(float_arr, [2,4], 'DT_FLOAT' )))).to eq(float_arr)
      expect(tr(instance.parse_value(get_value(int_arr, [2,4], 'DT_INT32' )))).to eq(int_arr)
    end

    specify "convets scaler tensor value to ruby" do
      expect(instance.parse_value(get_scalar_value(1.0, 'DT_FLOAT'))).to eq(1.0)
      expect(instance.parse_value(get_scalar_value(2, 'DT_INT32'))).to eq(2)
    end
  end
end