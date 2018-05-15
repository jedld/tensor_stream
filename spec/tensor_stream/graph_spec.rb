require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe TensorStream::Graph do
  let(:tf) { TensorStream }

  describe ".named_scope" do
    it "creates named scopes" do
      tf.graph.as_default do |g|
        c = tf.constant(5.0, name: "c")
        expect(c.name).to eq("c")
        c_1 = tf.constant(6.0, name: "c")
        expect(c_1.name).to eq("c_1")

        g.name_scope("nested") do |scope|
          nested_c = tf.constant(10.0, name: "c")
          expect(nested_c.name).to eq("nested/c")

          g.name_scope("inner") do
            nested_inner_c = tf.constant(20.0, name: "c")
            expect(nested_inner_c.name).to eq("nested/inner/c")
            expect(g.get_name_scope).to eq("nested/inner")
          end

          nested_d = tf.constant(10.0, name: "d")
          expect(nested_d.name).to eq("nested/d")
        end
      end
    end
  end
end