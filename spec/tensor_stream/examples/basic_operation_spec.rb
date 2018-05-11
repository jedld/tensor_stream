require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe "Illustrates basic tensorstream operations" do
  it "performs matrix multiplication" do
    # Basic constant operations
    # The value returned by the constructor represents the output
    # of the Constant op.
    a = TensorStream.constant(2)
    b = TensorStream.constant(3)

    # Launch the default graph.
    TensorStream.Session do |sess|
        puts("a=2, b=3")
        puts("Addition with constants: %i" % sess.run(a+b))
        expect(sess.run(a+b)).to eq(5.0)
        puts("Multiplication with constants: %i" % sess.run(a*b))
        expect(sess.run(a*b)).to eq(6.0)

        # Basic Operations with variable as graph input
        # The value returned by the constructor represents the output
        # of the Variable op. (define as input when running session)
        # tf Graph input
        a = TensorStream.placeholder(:int16)
        b = TensorStream.placeholder(:int16)

        # Define some operations
        add = TensorStream.add(a, b)
        mul = TensorStream.multiply(a, b)

         # Launch the default graph.
        TensorStream.Session() do |sess|
          # Run every operation with variable input
          puts("Addition with variables: %i" % sess.run(add, feed_dict: {a => 2, b => 3}))
          puts("Multiplication with variables: %i" % sess.run(mul, feed_dict: {a => 2, b => 3}))


          # ----------------
          # More in details:
          # Matrix Multiplication from TensorFlow official tutorial

          # Create a Constant op that produces a 1x2 matrix.  The op is
          # added as a node to the default graph.
          #
          # The value returned by the constructor represents the output
          # of the Constant op.
          matrix1 = TensorStream.constant([[3.0, 3.0]])

          # Create another Constant that produces a 2x1 matrix.
          matrix2 = TensorStream.constant([[2.0],[2.0]])

          # Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
          # The returned value, 'product', represents the result of the matrix
          # multiplication.
          product = TensorStream.matmul(matrix1, matrix2)

          # To run the matmul op we call the session 'run()' method, passing 'product'
          # which represents the output of the matmul op.  This indicates to the call
          # that we want to get the output of the matmul op back.
          #
          # All inputs needed by the op are run automatically by the session.  They
          # typically are run in parallel.
          #
          # The call 'run(product)' thus causes the execution of threes ops in the
          # graph: the two constants and matmul.
          #
          # The output of the op is returned in 'result' as a numpy `ndarray` object.
          TensorStream.Session do |sess|
            result = sess.run(product)
            expect(result).to eq([[12.0]])
          end
        end
      end
  end
end