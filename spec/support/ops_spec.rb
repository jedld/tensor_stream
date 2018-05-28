RSpec.shared_examples "standard ops evaluator" do
  context ".argmin" do
    it "Returns the index with the smallest value across axes of a tensor. " do
      a = tf.constant([
        [31, 23,  4, 24, 27, 34],
        [18,  3, 25,  0,  6, 35],
        [28, 14, 33, 22, 20,  8],
        [13, 30, 21, 19,  7,  9],
        [16,  1, 26, 32,  2, 29],
        [17, 12,  5, 11, 10, 15]])

      b = tf.constant([1,2,3,4,5,6])
      expect(sess.run(tf.argmin(a))).to eq([3, 4, 0, 1, 4, 2])
    end
  end

  context '.argmax' do
    it 'Returns the index with the largest value across axes of a tensor. (deprecated arguments)' do
      a = tf.constant([
        [31, 23,  4, 24, 27, 34],
        [18,  3, 25,  0,  6, 35],
        [28, 14, 33, 22, 20,  8],
        [13, 30, 21, 19,  7,  9],
        [16,  1, 26, 32,  2, 29],
        [17, 12,  5, 11, 10, 15]])

      b = tf.constant([1,2,3,4,5,6])
      expect(sess.run(tf.argmax(a))).to eq([0, 3, 2, 4, 0, 1])
      expect(sess.run(tf.argmax(a, 1))).to eq([5, 5, 2, 1, 3, 0])
      expect(sess.run(tf.argmax(a, 0))).to eq([0, 3, 2, 4, 0, 1])
      expect(sess.run(tf.argmax(b, 0))).to eq(5)
      expect(sess.run(tf.argmax(b, 0, output_type: :float32))).to eql(5.0)
    end
  end

  context ".add" do
    it "adds 2 tensors element-wise" do
      a = tf.constant(1.0)
      b = tf.constant(2.0)
      expect(sess.run(tf.add(a, b))).to eq(3.0)

      a = tf.constant([1.0, 1.1])
      b = tf.constant([2.0, 1.5])
      expect(sess.run(tf.add(a, b))).to eq([3.0, 2.6])
    end

    specify "supports broadcasting" do
      a = tf.constant([1.0, 1.1])
      b = tf.constant(2.0)
      expect(sess.run(tf.add(a, b))).to eq([3.0, 3.1])
    end
  end

  context ".sub" do
    let(:a) { tf.constant([1.0, 2.0, 3.0])}
    let(:b) { tf.constant([0.1, 0.2, 0.3])}
    let(:c) { tf.constant(0.1) }
    let(:m) { tf.constant([[1.0, 2.0, 3.0], [2.0, 3.0 ,4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]) }

    it "substracts two arrays" do
      expect(sess.run((a - b))).to eq([0.9, 1.8, 2.7])
    end

    it "substracts an array and a constant" do
      expect(sess.run((a - c))).to eq([0.9, 1.9, 2.9])
    end

    it "substracts a matrix and an array" do
      expect(sess.run((m - a))).to eq([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    end

    specify "gradients" do
      expect(sess.run(tf.gradients(a - b, [a,b]))).to eq([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    end
  end
end