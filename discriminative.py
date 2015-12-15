import tensorflow as tf
import numpy

INPUT_SIZE = 1
HIDDEN_UNIT_SIZE = 64
TRAIN_DATA_SIZE = 100

input = 0.5 + numpy.random.random(TRAIN_DATA_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE]) * 0.5
# ~U(0.5, 1.0)

def inference(input, hidden1_weight, hidden1_bias, output_weight, output_bias):
  hidden1_output = tf.nn.relu(tf.matmul(input, hidden1_weight) + hidden1_bias)
  output = tf.sigmoid(tf.matmul(hidden1_output, output_weight) + output_bias)
  return output

def trainable_inference(input):
  hidden1_weight = tf.get_variable(
    "d_hidden1_weight",
    [INPUT_SIZE, HIDDEN_UNIT_SIZE],
    initializer = tf.random_normal_initializer(0, 0.1)
  )
  hidden1_bias = tf.get_variable(
    "d_hidden1_bias",
    [HIDDEN_UNIT_SIZE],
    initializer = tf.constant_initializer(0.1)
  )
  output_weight = tf.get_variable(
    "d_output_weight",
    [HIDDEN_UNIT_SIZE, 1],
    initializer = tf.random_normal_initializer(0, 0.1)
  )
  output_bias = tf.get_variable(
    "d_output_bias",
    [1],
    initializer = tf.constant_initializer(0.1)
  )
  return inference(input, hidden1_weight, hidden1_bias, output_weight, output_bias)

def get_train_params():
  hidden1_weight = tf.get_variable("d_hidden1_weight", [INPUT_SIZE, HIDDEN_UNIT_SIZE])
  hidden1_bias = tf.get_variable("d_hidden1_bias", [HIDDEN_UNIT_SIZE])
  output_weight = tf.get_variable("d_output_weight", [HIDDEN_UNIT_SIZE, 1])
  output_bias = tf.get_variable("d_output_bias", [1])
  return [hidden1_weight, hidden1_bias, output_weight, output_bias]

def loss(output_from_given_data, output_from_noise):
  with tf.name_scope('d_loss') as scope:
    loss_1 = tf.reduce_sum(tf.log(output_from_given_data)) 
    loss_2 = tf.reduce_sum(tf.log(1 - output_from_noise))
  return [loss_1, loss_2]

