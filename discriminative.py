import tensorflow as tf
import numpy

INPUT_SIZE = 1
HIDDEN_UNIT_SIZE = 2
TRAIN_DATA_SIZE = 4

input = numpy.random.random(TRAIN_DATA_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE]) * 3
# ~U(0, 1)

def inference(input_placeholder):
  with tf.name_scope('d_hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="d_hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="d_hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(input_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('d_output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 1], stddev=0.1), name="d_output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[1]), name="d_output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return tf.sigmoid(output)

def loss(output_from_given_data, output_from_noise):
  with tf.name_scope('d_loss') as scope:
    loss = tf.reduce_sum(tf.log(output_from_given_data)) + tf.reduce_sum(tf.log(1 - output_from_noise))
  return loss

def loss_(output, given_data_label_placeholder):
  with tf.name_scope('d_loss') as scope:
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, given_data_label_placeholder))
    tf.scalar_summary('d_loss', loss)
  return loss

