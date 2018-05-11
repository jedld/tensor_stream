#!/usr/bin/env ruby

require 'brains'

# This neural network will predict the species of an iris based on sepal and petal size
# Dataset: http://en.wikipedia.org/wiki/Iris_flower_data_set

rows = File.readlines("iris.data").map {|l| l.chomp.split(',') }

rows.shuffle!

label_encodings = {
  "Iris-setosa"     => [1, 0, 0],
  "Iris-versicolor" => [0, 1, 0],
  "Iris-virginica"  => [0, 0 ,1]
}

x_data = rows.map {|row| row[0,4].map(&:to_f) }
y_data = rows.map {|row| label_encodings[row[4]] }

# Normalize data values before feeding into network
normalize = -> (val, high, low) {  (val - low) / (high - low) } # maps input to float between 0 and 1

columns = (0..3).map do |i|
  x_data.map {|row| row[i] }
end

x_data.map! do |row|
  row.map.with_index do |val, j|
    max, min = columns[j].max, columns[j].min
    normalize.(val, max, min)
  end
end

x_train = x_data.slice(0, 100)
y_train = y_data.slice(0, 100)

x_test = x_data.slice(100, 50)
y_test = y_data.slice(100, 50)

test_cases = []
x_train.each_with_index do |x, index|
  test_cases << [x, y_train[index] ]
end

validation_cases = []
x_test.each_with_index do |x, index|
  test_cases << [x, y_test[index] ]
end



# Build a 3 layer network: 4 input neurons, 4 hidden neurons, 3 output neurons
# Bias neurons are automatically added to input + hidden layers; no need to specify these
nn = Brains::Net.create(4, 3, 1, { neurons_per_layer: 4 })
nn.randomize_weights

prediction_success = -> (actual, ideal) {
  predicted = (0..2).max_by {|i| actual[i] }
  ideal[predicted] == 1
}

mse = -> (actual, ideal) {
  errors = actual.zip(ideal).map {|a, i| a - i }
  (errors.inject(0) {|sum, err| sum += err**2}) / errors.length.to_f
}

error_rate = -> (errors, total) { ((errors / total.to_f) * 100).round }

run_test = -> (nn, inputs, expected_outputs) {
  success, failure, errsum = 0,0,0
  inputs.each.with_index do |input, i|
    output = nn.feed input
    prediction_success.(output, expected_outputs[i]) ? success += 1 : failure += 1
    errsum += mse.(output, expected_outputs[i])
  end
  [success, failure, errsum / inputs.length.to_f]
}

puts "Testing the untrained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Untrained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"


puts "\nTraining the network...\n\n"

t1 = Time.now
result = nn.optimize(test_cases, 0.01, 1_000 ) { |i, error|
  puts "#{i} #{error}"
}

# puts result
puts "\nDone training the network: #{result[:iterations]} iterations, #{(result[:error] * 100).round(2)}% mse, #{(Time.now - t1).round(1)}s"


puts "\nTesting the trained network..."

success, failure, avg_mse = run_test.(nn, x_test, y_test)

puts "Trained classification success: #{success}, failure: #{failure} (classification error: #{error_rate.(failure, x_test.length)}%, mse: #{(avg_mse * 100).round(2)}%)"
