require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::MathGradientsForward do
  include TensorStream::OpHelper
  let(:tf) { TensorStream }
  let(:sess) { tf.session }

  context "derivative computation" do
    specify "drt is equal to tensor" do
      a = tf.constant([1.0, 1.0])
      g = tf.gradients(a, [a])
      expect(sess.run(g)).to eq([[1.0, 1.0]])
    end

    context "add" do
      specify do
        a = tf.constant([1.0, 1.0])
        b = tf.constant([2.0, 1.1])

        f = a + b
        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[1.0, 1.0], [1.0, 1.0]])

        f = a + a + b

        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[2.0, 2.0], [1.0, 1.0]])
      end

      it "handles shape differences, rank 2 vs 0" do
        a = tf.constant([[1, 2],[3, 4],[5, 6]])
        b = tf.constant(1)
        sum = a + b
        g = tf.gradients(sum, [a, b])

        expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], 6])
      end
    end

    context "sub" do
      specify do
        a = tf.constant([1.0, 1.0])
        b = tf.constant([2.0, 1.1])

        f = a - b
        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[1.0, 1.0], [-1.0, -1.0]])

        f = a - a - b

        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[0.0, 0.0], [-1.0, -1.0]])
      end

      it "handles shape differences, rank 2 vs 0" do
        a = tf.constant([[1, 2],[3, 4],[5, 6]])
        b = tf.constant(1)
        sum = a - b
        g = tf.gradients(sum, [a, b])
        sess.run(g)
        expect(sess.run(g)).to eq([[[1, 1], [1, 1], [1, 1]], -6])
      end

    end

    specify "multiplication" do
      a = tf.constant([1.0, 1.0])
      b = tf.constant([2.0, 1.1])

      f = a * b
      g = tf.gradients(f, [a, b])
      expect(sess.run(g)).to eq([[2.0, 1.1], [1.0, 1.0]])

      f = cons(2) * a + cons(3) * b
      g = tf.gradients(f, [a, b])
      expect(sess.run(g)).to eq([[2.0, 2.0], [3.0, 3.0]])
    end

    context "div" do
      specify do
        a = tf.constant(2.5)
        b = tf.constant(3.1)
        f = a/b
        g = tf.gradients(f, [a, b])
        expect(tr(sess.run(g))).to eq([0.3226, -0.2601])
      end
    end

    context "pow" do
      specify do
        x = tf.constant([[1.1, 1.3], [1.3, 1.2]])
        y = tf.constant([[1.5, 2.0], [1.1, 2.0]])
        p = tf.pow(x, y)  # [[256, 65536], [9, 27]]
        g = tf.gradients(p, [x, y])
        expect(tr(sess.run(g))).to eq([
          [[1.5732, 2.6], [1.1292, 2.4]],
          [[0.11, 0.4434], [0.3501, 0.2625]]
        ])
      end
    end

    context "matrix multiplication" do
      specify do
        y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
        x = tf.constant([[4.0, 5.0], [5.0, 6.0]], dtype: :float32)

        c = tf.matmul(x, y)

        expect(c.eval).to eq([[19, 28], [23, 34]])
        c_grad = tf.gradients(c, [x, y])
        expect(c_grad.eval).to eq([
          [[3.0, 7.0], [3.0, 7.0]],
          [[9.0, 9.0], [11.0, 11.0]]
        ])
      end

      specify do
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])

        y = tf.matmul(a, tf.sin(b))
        expect(tr(y.eval)).to eq([[-2.0631, -4.0106, -2.2707], [-3.3563, -7.0425, -4.2538]])
        g = tf.gradients(y, [a, b])
        expect(tr(g.eval)).to eq([[[2.0585, -2.0806, -2.0806], [2.0585, -2.0806, -2.0806]], [[3.7695, -0.7275, -4.5557], [-5.8735, 0.031, 5.907], [-7.5516, 0.0398, 7.5947]]])
      end
    end
  end
end