module SupportedOp
  def supported_op(op, &block)
    op = op.gsub('.','')
    if described_class.ops.keys.include?(op.to_sym)
      context(".#{op}", &block)
    end
  end
end