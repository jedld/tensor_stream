require "bundler/setup"
require "tensor_stream"
require 'awesome_print'
# require 'pry-byebug'

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end

# Helper function to truncate floating point values (for testing)
# truncation is done in tests since different machines return the last digits of
# fp values differently
def tr(t, places = 4)
  if t.is_a?(Array)
    return t.collect do |v|
      tr(v)
    end
  end

  return t unless t.kind_of?(Float)

  t.round(places)
end
