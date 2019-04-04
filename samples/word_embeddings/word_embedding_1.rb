#
# A ruby port of https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer by Guillaume Chevalier
#
# This is a port so some weird python like conventions may have been left behind
require "bundler/setup"
require "tensor_stream"
require "chakin-rb/chakin"
# require 'pry-byebug'
require 'zip'

tf = TensorStream

CHAKIN_INDEX = 17
NUMBER_OF_DIMENSIONS = 25
SUBFOLDER_NAME = "glove.twitter.27B"

DATA_FOLDER = "embeddings"
ZIP_FILE = File.join(DATA_FOLDER, "#{SUBFOLDER_NAME}.zip")
ZIP_FILE_ALT = "glove" + ZIP_FILE[5..nil]  # sometimes it's lowercase only...
UNZIP_FOLDER = File.join(DATA_FOLDER, SUBFOLDER_NAME)

if SUBFOLDER_NAME[-1] == "d"
  GLOVE_FILENAME = File.join(UNZIP_FOLDER, "#{SUBFOLDER_NAME}.txt")
else
  GLOVE_FILENAME = File.join(UNZIP_FOLDER, "#{SUBFOLDER_NAME}.#{NUMBER_OF_DIMENSIONS}d.txt")
end

if !File.exist?(ZIP_FILE) && !File.exist?(UNZIP_FOLDER)
  # GloVe by Stanford is licensed Apache 2.0:
  #     https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
  #     http://nlp.stanford.edu/data/glove.twitter.27B.zip
  #     Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
  puts "Downloading embeddings to '#{ZIP_FILE}'"
  Chakin::Vectors.download(number: CHAKIN_INDEX, save_dir: "./#{DATA_FOLDER}")
else
  puts "Embeddings already downloaded."
end

if !File.exists?(UNZIP_FOLDER)
  if !File.exists?(ZIP_FILE) && !File.exists?(ZIP_FILE_ALT)
    ZIP_FILE = ZIP_FILE_ALT
  end
  FileUtils.mkdir_p(UNZIP_FOLDER)
  Zip::File.open(ZIP_FILE) do |zipfile|
    zipfile.each do |file|
      puts "Extracting embeddings to '#{UNZIP_FOLDER}/#{file.name}'"
      fpath = File.join(UNZIP_FOLDER, file.name)
      zipfile.extract(file, fpath) unless File.exist?(fpath)
    end
  end
else
  puts "Embeddings already extracted."
end

##
#   Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
#   `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
#   `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
def load_embedding_from_disks(glove_filename, with_indexes: true)
  word_to_index_dict = {}
  index_to_embedding_array = []
  word_to_embedding_dict = {}
  representation = nil

  last_index = nil
  File.open(glove_filename, 'r').each_with_index do |line, i|
    split = line.split(' ')

    word = split.shift

    representation = split
    representation.map! { |val| val.to_f }

    if with_indexes
      word_to_index_dict[word] = i
      index_to_embedding_array << representation
    else
      word_to_embedding_dict[word] = representation
    end
    last_index = i
  end

  _WORD_NOT_FOUND = [0.0] * representation.size  # Empty representation for unknown words.
  if with_indexes
    _LAST_INDEX = last_index + 1
    word_to_index_dict = Hash.new(_LAST_INDEX).merge(word_to_index_dict)
    index_to_embedding_array = index_to_embedding_array + [_WORD_NOT_FOUND]
    return word_to_index_dict, index_to_embedding_array
  else
    word_to_embedding_dict = Hash.new(_WORD_NOT_FOUND)
    return word_to_embedding_dict
  end
end

puts "Loading embedding from disks..."
word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes: true)
puts "Embedding loaded from disks."

vocab_size, embedding_dim = index_to_embedding.shape
puts "Embedding is of shape: #{index_to_embedding.shape}"
puts "This means (number of words, number of dimensions per word)"
puts "The first words are words that tend occur more often."

puts "Note: for unknown words, the representation is an empty vector,\n" +
      "and the index is the last one. The dictionnary has a limit:"
puts "    \"A word\" --> \"Index in embedding\" --> \"Representation\""
word = "worsdfkljsdf"
idx = word_to_index[word]
embd = index_to_embedding[idx].map { |v| v.to_i }  # "int" for compact print only.
puts "    #{word} --> #{idx} --> #{embd}"
word = "the"
idx = word_to_index[word]
embd = index_to_embedding[idx]  # "int" for compact print only.
puts "    #{word} --> #{idx} --> #{embd}"

words = [
  "The", "Teh", "A", "It", "Its", "Bacon", "Star", "Clone", "Bonjour", "Intelligence",
  "À", "A", "Ça", "Ca", "Été", "C'est", "Aujourd'hui", "Aujourd", "'", "hui", "?", "!", ",", ".", "-", "/", "~"
]

words.each do |word|
  word_ = word.downcase
  embedding = index_to_embedding[word_to_index[word_]]
  norm = Vector::elements(embedding).norm
  puts (word + ": ").ljust(15) + norm.to_s
end

puts "Note: here we printed words starting with capital letters, \n" +
"however to take their embeddings we need their lowercase version (str.downcase)"

batch_size = nil  # Any size is accepted

tf.reset_default_graph
sess =  tf.session

# Define the variable that will hold the embedding:
tf_embedding = tf.variable(
    tf.constant(0.0, shape: index_to_embedding.shape),
    trainable: false,
    name: "Embedding"
)

tf_word_ids = tf.placeholder(:int32, shape: [batch_size])

tf_word_representation_layer = tf.nn.embedding_lookup(tf_embedding, tf_word_ids)

tf_embedding_placeholder = tf.placeholder(:float32, shape: index_to_embedding.shape)
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)

sess.run(
    tf_embedding_init,
    feed_dict: {
        tf_embedding_placeholder => index_to_embedding
    }
)

puts "Embedding now stored in TensorStream. Can delete ruby array to clear some CPU RAM."

batch_of_words = ["Hello", "World", "!"]
batch_indexes = batch_of_words.map { |w| word_to_index[w.downcase] }

embedding_from_batch_lookup = sess.run(
    tf_word_representation_layer,
    feed_dict: {
        tf_word_ids => batch_indexes
    }
)

puts "Representations for #{batch_of_words}:"
puts embedding_from_batch_lookup.inspect

prefix = SUBFOLDER_NAME + "." + NUMBER_OF_DIMENSIONS.to_s + "d"
TF_EMBEDDINGS_FILE_NAME = File.join(DATA_FOLDER, prefix + ".ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = File.join(DATA_FOLDER, prefix + ".json")

variables_to_save = [tf_embedding]
embedding_saver = tf::Train::Saver.new(variables_to_save)
embedding_saver.save(sess, TF_EMBEDDINGS_FILE_NAME)
puts "TF embeddings saved to '#{TF_EMBEDDINGS_FILE_NAME}'."

sess.close

File.open(DICT_WORD_TO_INDEX_FILE_NAME, 'w') do |f|
  f.write(word_to_index.to_json)
end
puts "word_to_index dict saved to '#{DICT_WORD_TO_INDEX_FILE_NAME}'."

words_B = "like absolutely crazy not hate bag sand rock soap"
r = words_B.split.map { |w| word_to_index[w.strip()] }
puts words_B
puts r.inspect
puts "done"