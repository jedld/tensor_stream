require "spec_helper"

RSpec.describe "Tests on RNN cells" do
  let(:tf) { TensorStream }
  let(:sess) { tf.session }

  before(:each) do
    TensorStream::Tensor.reset_counters
    # TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  context "Creating an GRUCell layer" do
    specify do
      num_units = 200
      num_layers = 3
      cells = num_layers.times.map  { TensorStream::RNN::GRUCell.new(num_units) }
      cell = TensorStream::RNN::MultiRNNCell.new(cells)
      data = TensorStream.placeholder(:float32, shape: [nil, nil, 4])

      output, state = TensorStream.nn.dynamic_rnn(cell, data, dtype: :float32)
      init = tf.global_variables_initializer
      sess.run(init)
      expect(sess.run(output, feed_dict: { data => [[[0.6509, 0.6647, 0.3442, 0.6021, 0.976, 0.095583, 0.0592, 0.16, 0.13, 0.68, 0.71, 0.9, 0.915, 0.20, 0.173, 0.381, 0.44, 0.12, 0.92, 0.883, 0.134, 0.19, 0.57, 0.8, 0.79, 0.744, 0.18, 0.998]]]})).to eq([[[-4.3309941e-03, -2.9013143e-03,  3.2055620e-04,  2.7545078e-03,
        2.4328143e-03,  3.2274169e-04,  9.3601068e-04,  1.8309194e-04,
       -1.8306441e-03,  4.2525064e-03, -3.8904264e-03,  2.4813884e-03,
        6.9953792e-04, -1.0377092e-03,  1.4904332e-04, -6.9280881e-03,
       -6.8249023e-03, -4.7497889e-03,  1.7265507e-04,  1.1681904e-03,
        6.3028454e-04,  1.1375329e-03, -1.5935288e-03,  2.2703542e-03,
       -2.1326453e-03,  8.1180857e-04, -8.0484297e-04,  3.4322945e-04,
        4.2483601e-04,  3.2341292e-03,  1.1759397e-03, -4.7016204e-03,
       -1.2294233e-03, -5.8067329e-03, -5.3632807e-04, -1.7071337e-03,
       -2.8053620e-03,  1.0144595e-03, -2.7681582e-03,  2.8055587e-03,
       -2.4332574e-03,  1.4578257e-03,  1.8566524e-04, -1.8588592e-04,
       -1.8093714e-03,  1.1900129e-03,  8.7442459e-04, -2.0703129e-04,
       -1.4657243e-03,  7.2640867e-04,  1.2673503e-03,  4.1001984e-03,
       -1.0834703e-03, -2.2974994e-03,  7.9513079e-04,  4.0807952e-03,
        3.6326977e-03, -3.0138283e-03, -1.3227999e-03, -9.6355245e-04,
        4.2765606e-03, -1.1180849e-03, -2.6499890e-03,  3.1052867e-03,
        1.4058941e-03,  3.1453483e-03, -1.1338795e-03, -1.5314255e-04,
        1.1022390e-03,  1.4409699e-03, -1.5406676e-03,  4.4998699e-03,
       -1.7282064e-03, -2.3454458e-03, -1.6983300e-03, -5.6600696e-03,
        3.7869874e-03,  1.6803008e-03,  2.2678030e-05,  3.3588128e-03,
       -7.8350464e-03,  2.7052218e-03, -5.2165599e-03, -5.6056059e-03,
        1.9817827e-04, -3.2416217e-03,  1.4316471e-03, -7.0043537e-04,
       -1.3276114e-03,  1.9941796e-03,  1.1670011e-03,  1.1665080e-03,
        2.4173055e-03,  9.2125015e-04, -3.9950539e-03, -6.5900744e-03,
       -1.1591503e-03,  2.4368097e-03, -8.7890803e-04, -1.5462245e-03,
       -3.0545838e-04, -7.6529048e-03,  1.5787436e-03, -3.7125971e-03,
        2.1181889e-03, -3.6693672e-03, -8.1122480e-04,  6.0073112e-04,
       -1.5408505e-03, -6.7395886e-04,  1.1172977e-03,  3.1569162e-03,
       -5.1317108e-03, -5.3451122e-03,  5.0359080e-04,  2.7534461e-06,
        3.1056220e-04, -3.0604232e-04, -4.3606316e-03, -3.6267962e-03,
        7.5252930e-04, -4.7285943e-03, -5.0890362e-03,  1.4882973e-04,
        2.5786352e-03, -3.0439051e-03, -3.1728363e-03,  1.6596986e-04,
        1.5376740e-04,  2.2468187e-03,  5.1866504e-03, -1.9066201e-03,
        4.1128406e-03, -8.4361783e-04, -1.2664661e-04,  2.6693465e-03,
        8.0641330e-04, -6.1958693e-03, -1.7020332e-03, -2.6903262e-03,
       -2.4073668e-04,  5.2841855e-03, -1.8296873e-03,  9.7188173e-04,
       -6.2529562e-04, -6.1988208e-04, -1.8761125e-03,  3.0283572e-04,
       -6.0539396e-04,  1.1641845e-03,  3.2193350e-04, -1.9536566e-03,
       -3.0688175e-03,  6.3443827e-03, -1.5514240e-03, -2.2228630e-03,
       -2.4026013e-03, -5.1612446e-05,  5.0473819e-03,  7.8599704e-03,
        3.0000132e-04,  1.9277779e-03,  1.9351617e-03, -2.2193249e-03,
       -3.3007327e-03, -9.5050718e-04, -3.0882966e-03,  2.9457984e-03,
       -3.5070083e-03,  7.6522434e-04, -1.9678259e-03,  2.9292081e-03,
        4.9761864e-03,  1.7404162e-03, -1.1906663e-03, -1.7227083e-04,
        2.1612929e-04, -1.3946920e-03, -3.2287610e-03, -1.3145164e-03,
        2.1840071e-03,  2.3042371e-03, -2.1751630e-03,  4.8385668e-03,
        3.0477126e-03,  3.0019616e-03, -3.2609154e-03, -4.0550609e-03,
       -7.5542775e-04, -3.7181093e-03,  5.8597289e-03, -3.6335310e-03,
        7.5673657e-03,  2.2943750e-04,  5.2700010e-03,  6.2028547e-03,
        7.3801930e-04,  8.1790955e-04, -1.8336291e-04,  2.5510581e-03]]])
    end
  end
end