import tensorflow as tf
import numpy

INPUT_SIZE = 2
HIDDEN_UNIT_SIZE = 2
TRAIN_DATA_SIZE = 4
OUTPUT_SIZE = 1

input = numpy.random.random(TRAIN_DATA_SIZE * INPUT_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE])

def inference(input_placeholder):
  with tf.name_scope('g_hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="g_hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="g_hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(input_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('g_output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, OUTPUT_SIZE], stddev=0.1), name="g_output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE]), name="g_output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return tf.nn.relu(output)

input_placeholder = tf.placeholder("float", [None, INPUT_SIZE], name="g_input_placeholder")
output = inference(input_placeholder)

def feed():
  return {g_input_placeholder: input_placeholder}

