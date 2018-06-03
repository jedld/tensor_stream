require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Operation do

  let(:tf) { TensorStream } # allow calls to look like tensorflow
  let(:sess) { tf.session }

end