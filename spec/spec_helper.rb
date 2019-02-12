require "bundler/setup"
require "simplecov"
SimpleCov.start
require 'pry-byebug'
require "tensor_stream"
require "awesome_print"


Dir["./spec/support/**/*.rb"].sort.each { |f| require f }

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  config.before(:each) do
    TensorStream::Tensor.reset_counters
    # TensorStream::Operation.reset_counters
    TensorStream.reset_default_graph
  end
end

# Helper function to truncate floating point values (for testing)
# truncation is done in tests since different machines return the last digits of
# fp values differently
def tr(t, places = 4)
  if t.is_a?(Array)
    return t.collect do |v|
      tr(v, places)
    end
  end

  return t unless t.is_a?(Float)

  t.round(places)
end

def trf(t, places)
  if t.is_a?(Array)
    return t.collect do |v|
      trf(v, places)
    end
  end

  return t unless t.is_a?(Float)
  t.truncate(places)
end
