import tensorflow as tf
import numpy

INPUT_SIZE = 1
HIDDEN_UNIT_SIZE = 2
TRAIN_DATA_SIZE = 4

input = numpy.random.random(TRAIN_DATA_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE]) * 3
# ~U(0, 1)

def given_data_probability(input):
  if (input < 1): return 1
  else: return 0

given_data_label_true = numpy.vectorize(given_data_probability)(input).flatten()
given_data_label = numpy.array([given_data_label_true, 1 - given_data_label_true]).transpose()
input_placeholder = tf.placeholder("float", [None, INPUT_SIZE], name="d_input_placeholder")

def inference(input_placeholder):
  with tf.name_scope('d_hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="d_hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="d_hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(input_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('d_output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 2], stddev=0.1), name="d_output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[2]), name="d_output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return output

def loss(d_output, g_output, given_data_label_placeholder):
  # FIXME
  return 1


def loss_(output, given_data_label_placeholder):
  with tf.name_scope('d_loss') as scope:
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, given_data_label_placeholder))
    tf.scalar_summary('d_loss', loss)
  return loss

def train_var_list():
  return [
    tf.get_variable('d_hidden1/d_hidden1_weight', [INPUT_SIZE, HIDDEN_UNIT_SIZE]),
    tf.get_variable('d_hidden1/d_hidden1_bias', [HIDDEN_UNIT_SIZE]),
    tf.get_variable('d_output/d_output_weight', [HIDDEN_UNIT_SIZE, 2]),
    tf.get_variable('d_output/d_output_bias', [2])
  ]

# def training(loss):
#   with tf.name_scope('d_training') as scope:
#     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#   return train_step

output = inference(input_placeholder)

