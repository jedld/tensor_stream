TensorStream::OpMaker.define_operation :const do
  exclude!

  define_constant do |node, _partial|
    node.options[:value]
  end
end