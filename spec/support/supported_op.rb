module SupportedOp
  def supported_op(op, &block)
    op = op.is_a?(Symbol) ? op.to_s : op
    op = op.delete(".")
    if described_class.ops.key?(op.to_sym)
      context(".#{op}", &block)
    end
  end
end
