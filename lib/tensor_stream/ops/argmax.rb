TensorStream::OpMaker.define_operation :argmax do |op|
  what_it_does "Returns the index with the largest value across axes of a tensor."

  parameter :input_a, "tensor X", validate: 'NUMERIC_TYPES'
  parameter :axis, "Describes which axis of the input tensor to reduce across. For vectors, use axis = 0", :nil, validate: 'INTEGER_TYPES'

  option :name, "Optional name", :nil
  option :dimension, "Same as axis", :nil
  option :output_type, "Output data type defaults to int32", ":int32"

  define_gradient do |grad, node, params|
    [nil, nil]
  end
end