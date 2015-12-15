import tensorflow as tf
import numpy

INPUT_SIZE = 1
HIDDEN_UNIT_SIZE = 64
TRAIN_DATA_SIZE = 100
OUTPUT_SIZE = 1

input = -3 + numpy.random.random(TRAIN_DATA_SIZE * INPUT_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE]) * 6

def inference(input_placeholder):
  with tf.name_scope('g_hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_UNIT_SIZE], stddev=1.0), name="g_hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.01, shape=[HIDDEN_UNIT_SIZE]), name="g_hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(input_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('g_output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, OUTPUT_SIZE], stddev=1.5), name="g_output_weight")
    output_bias = tf.Variable(tf.constant(0.01, shape=[OUTPUT_SIZE]), name="g_output_bias")
    output = tf.sigmoid(tf.matmul(hidden1_output, output_weight) + output_bias)
  return output

