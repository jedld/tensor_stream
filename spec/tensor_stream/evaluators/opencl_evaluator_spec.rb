require "spec_helper"
require 'benchmark'
require 'tensor_stream/evaluator/opencl_evaluator'

RSpec.describe TensorStream::Evaluator::OpenclEvaluator do
  let(:tf) { TensorStream }
  let(:sess) { TensorStream.session(:opencl_evaluator) }
  let(:instance) { described_class.new(sess, {})}

  it_behaves_like "standard ops evaluator"

  def create_session
    TensorStream.session(:opencl_evaluator)
  end
end