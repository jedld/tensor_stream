# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'tensor_stream/version'

Gem::Specification.new do |spec|
  spec.name          = "tensor_stream"
  spec.version       = TensorStream::VERSION
  spec.authors       = ["Joseph Emmanuel Dayo"]
  spec.email         = ["joseph.dayo@gmail.com"]

  spec.summary       = %q{A Pure ruby tensorflow implementation}
  spec.description   = %q{A reimplementation of TensorFlow for ruby. This is a ground up implementation with no dependency on TensorFlow. Effort has been made to make the programming style as near to TensorFlow as possible, comes with a pure ruby evaluator by default as well with support for an opencl evaluator.}
  spec.homepage      = "http://www.github.com/jedld/tensor_stream"
  spec.license       = "MIT"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.
  if spec.respond_to?(:metadata)
    spec.metadata['allowed_push_host'] = "https://rubygems.org"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.14"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "awesome_print"
  spec.add_development_dependency "rubocop"
  if RUBY_ENGINE == 'ruby'
    spec.add_development_dependency "pry-byebug"
    spec.add_development_dependency "byepry"
  end
  spec.add_development_dependency "colorize"
  spec.add_development_dependency "rspec_junit_formatter"
  spec.add_development_dependency "mnist-learn"
  spec.add_development_dependency "simplecov"
  spec.add_development_dependency "standard"
  spec.add_dependency "deep_merge"
  spec.add_dependency "concurrent-ruby"
  spec.add_dependency "chunky_png"
end
