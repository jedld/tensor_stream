TensorStream::OpMaker.define_operation :sign do |op|
  op.what_it_does      "Computes sign of input element-wise."
  op.what_it_does_code "y = sign(x) = -1 if x < 0; 0 if x == 0 or tf.is_nan(x); 1 if x > 0."
  op.what_it_does      "Zero is returned for NaN inputs."

  op.parameter :input_a, "tensor X"
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    ts.zeros(ts.shape(params[0]), dtype: params[0].data_type)
  end
end