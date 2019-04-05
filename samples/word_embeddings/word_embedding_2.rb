#
# A ruby port of https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer by Guillaume Chevalier
#
# This is a port so some weird python like conventions may have been left behind
require "bundler/setup"
require "tensor_stream"
require "chakin-rb/chakin"
require 'pry-byebug'
require 'zip'

tf = TensorStream

batch_size = nil  # Any size is accepted
word_representations_dimensions = 25  # Embedding of size (vocab_len, nb_dimensions)


DATA_FOLDER = "embeddings"
SUBFOLDER_NAME = "glove.twitter.27B"
TF_EMBEDDING_FILE_NAME = "#{SUBFOLDER_NAME}.ckpt"
SUFFIX = SUBFOLDER_NAME + "." + word_representations_dimensions.to_s
TF_EMBEDDINGS_FILE_PATH = File.join(DATA_FOLDER, SUFFIX + "d.ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = File.join(DATA_FOLDER, SUFFIX + "d.json")

# Load a `word_to_index` dict mapping words to their id, with a default value
# of pointing to the last index when not found, which is the unknown word.
def load_word_to_index(dict_word_to_index_file_name)
  word_to_index = JSON.parse(File.read(dict_word_to_index_file_name))
  _LAST_INDEX = word_to_index.size - 1
  puts "word_to_index dict restored from '#{dict_word_to_index_file_name}'."
  word_to_index = Hash.new(_LAST_INDEX).merge(word_to_index)
  word_to_index
end

# """
# Define the embedding tf.Variable and load it.
# """
def load_embedding_tf(sess, word_to_index, tf_embeddings_file_path, nb_dims)

  # 1. Define the variable that will hold the embedding:
  tf_embedding = TensorStream.variable(
    TensorStream.constant(0.0, shape: [word_to_index.size-1, nb_dims]),
      trainable: false,
      name: "Embedding"
  )

  # 2. Restore the embedding from disks to TensorFlow, GPU (or CPU if GPU unavailable):
  variables_to_restore = [tf_embedding]
  embedding_saver = TensorStream::Train::Saver.new(variables_to_restore)
  embedding_saver.restore(sess, tf_embeddings_file_path)
  puts "TF embeddings restored from '#{tf_embeddings_file_path}'."

  tf_embedding
end


def cosine_similarity_tensorflow(tf_word_representation_A, tf_words_representation_B)
  """
  Returns the `cosine_similarity = cos(angle_between_a_and_b_in_space)`
  for the two word A to all the words B.
  The first input word must be a 1D Tensors (word_representation).
  The second input words must be 2D Tensors (batch_size, word_representation).
  The result is a tf tensor that must be fetched with `sess.run`.
  """
  a_normalized = TensorStream.nn.l2_normalize(tf_word_representation_A, axis: -1)
  b_normalized = TensorStream.nn.l2_normalize(tf_words_representation_B, axis: -1)
  TensorStream.reduce_sum(
      TensorStream.multiply(a_normalized, b_normalized),
      axis: -1
  )
end

# In case you didn't do the "%reset":
tf.reset_default_graph
sess = tf.session

# Load the embedding matrix in tf
word_to_index = load_word_to_index(
    DICT_WORD_TO_INDEX_FILE_NAME)
tf_embedding = load_embedding_tf(sess,
    word_to_index,
    TF_EMBEDDINGS_FILE_PATH,
    word_representations_dimensions)


# Input to the graph where word IDs can be sent in batch. Look at the "shape" args:
@tf_word_A_id = tf.placeholder(:int32, shape: [1])
@tf_words_B_ids = tf.placeholder(:int32, shape: [batch_size])

# Conversion of words to a representation
tf_word_representation_A = tf.nn.embedding_lookup(tf_embedding, @tf_word_A_id)
tf_words_representation_B = tf.nn.embedding_lookup(tf_embedding, @tf_words_B_ids)

# The graph output are the "cosine_similarities" which we want to fetch in sess.run(...).
@cosine_similarities = cosine_similarity_tensorflow(tf_word_representation_A, tf_words_representation_B)

print("Model created.")

# Note: there might be a better way to split sentences for GloVe.
# Please look at the documentation or open an issue to suggest a fix.
def sentence_to_word_ids(sentence, word_to_index)
  punctuation = ['.', '!', '?', ',', ':', ';', "'", '"', '(', ')']
  # Separating punctuation from words:
  punctuation.each do |punctuation_character|
    sentence.gsub!(punctuation_character, " #{punctuation_character} ")
  end
  # Removing double spaces and lowercasing:
  sentence = sentence.downcase.squeeze(" ").strip

  # Splitting on every space:
  split_sentence = sentence.split(" ")
  ids = split_sentence.map { |w| word_to_index[w.strip] }
  # Converting to IDs:
  ids = split_sentence.map { |w| word_to_index[w.strip] }
  [ids, split_sentence]
end

# Use the model in sess to predict cosine similarities.
def predict_cosine_similarities(sess, word_to_index, word_A, words_B)
  word_A_id, _ = sentence_to_word_ids(word_A, word_to_index)
  words_B_ids, split_sentence = sentence_to_word_ids(words_B, word_to_index)

  evaluated_cos_similarities = sess.run(
      @cosine_similarities,
      feed_dict: {
          @tf_word_A_id => word_A_id,
          @tf_words_B_ids => words_B_ids
      }
  )
  [evaluated_cos_similarities, split_sentence]
end

word_A = "Science"
words_B = "Hello internet, a vocano erupt like the bitcoin out of the blue and there is an unknownWord00!"

evaluated_cos_similarities, splitted = predict_cosine_similarities(sess, word_to_index, word_A, words_B)

puts "Cosine similarities with \"#{word_A}\":"
splitted.zip(evaluated_cos_similarities).each do |word, similarity|
  puts "    #{(word+":").ljust(15)}#{similarity}"
end

tf.reset_default_graph()


# Transpose word_to_index dict:
index_to_word = word_to_index.invert

# New graph
tf.reset_default_graph()
sess = tf.session

# Load the embedding matrix in tf
tf_word_to_index = load_word_to_index(
    DICT_WORD_TO_INDEX_FILE_NAME)

tf_embedding = load_embedding_tf(sess,
    tf_word_to_index,
    TF_EMBEDDINGS_FILE_PATH,
    word_representations_dimensions)

# An input word
tf_word_id = tf.placeholder(:int32, shape: [1])
tf_word_representation = tf.nn.embedding_lookup(tf_embedding, tf_word_id)

# An input
tf_nb_similar_words_to_get = tf.placeholder(:int32)

# Dot the word to every embedding
tf_all_cosine_similarities = cosine_similarity_tensorflow(
    tf_word_representation,
    tf_embedding)

# Getting the top cosine similarities.
tf_top_cosine_similarities, tf_top_word_indices = tf.top_k(
    tf_all_cosine_similarities,
    tf_nb_similar_words_to_get + 1,
    sorted: true
)

# Discard the first word because it's the input word itself:
tf_top_cosine_similarities = tf_top_cosine_similarities[1..nil]
tf_top_word_indices = tf_top_word_indices[1..nil]

# Get the top words' representations by fetching
# tf_top_words_representation = "tf_embedding[tf_top_word_indices]":
tf_top_words_representation = tf.gather(tf_embedding, tf_top_word_indices)

# Fetch 10 similar words:
nb_similar_words_to_get = 10


word = "king"
word_id = word_to_index[word]

top_cosine_similarities, top_word_indices, top_words_representation = sess.run(
    [tf_top_cosine_similarities, tf_top_word_indices, tf_top_words_representation],
    feed_dict: {
      tf_word_id => [word_id],
      tf_nb_similar_words_to_get => nb_similar_words_to_get
    }
)

puts "Top similar words to \"#{word}\":\n"
top_cosine_similarities.zip(top_word_indices).zip(top_words_representation).each do |w, word_repr|
  cos_sim, word_id = w
  puts "#{(index_to_word[word_id]+ ":").ljust(15)}#{(cos_sim.to_s + ",").ljust(15)}#{Vector::elements(word_repr).norm}"
end