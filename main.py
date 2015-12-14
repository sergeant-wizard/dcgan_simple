import tensorflow as tf
import numpy

INPUT_SIZE = 1
HIDDEN_UNIT_SIZE = 2
TRAIN_DATA_SIZE = 4

input = numpy.random.random(TRAIN_DATA_SIZE).reshape([TRAIN_DATA_SIZE, INPUT_SIZE]) * 3
# ~U(0, 3)

def given_data_probability(input):
  if (input < 1): return 1
  else: return 0

given_data_label_true = numpy.vectorize(given_data_probability)(input).flatten()
given_data_label = numpy.array([given_data_label_true, 1 - given_data_label_true]).transpose()

def inference(input_placeholder):
  with tf.name_scope('hidden1') as scope:
    hidden1_weight = tf.Variable(tf.truncated_normal([INPUT_SIZE, HIDDEN_UNIT_SIZE], stddev=0.1), name="hidden1_weight")
    hidden1_bias = tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNIT_SIZE]), name="hidden1_bias")
    hidden1_output = tf.nn.relu(tf.matmul(input_placeholder, hidden1_weight) + hidden1_bias)
  with tf.name_scope('output') as scope:
    output_weight = tf.Variable(tf.truncated_normal([HIDDEN_UNIT_SIZE, 2], stddev=0.1), name="output_weight")
    output_bias = tf.Variable(tf.constant(0.1, shape=[2]), name="output_bias")
    output = tf.matmul(hidden1_output, output_weight) + output_bias
  return output

def loss(output, given_data_label_placeholder):
  with tf.name_scope('loss') as scope:
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, given_data_label_placeholder))
    tf.scalar_summary('loss', loss)
  return loss

def training(loss):
  with tf.name_scope('training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

with tf.Graph().as_default():
  input_placeholder = tf.placeholder("float", [None, INPUT_SIZE], name="input_placeholder")
  given_data_label_placeholder = tf.placeholder("float", [None, 2], name="given_placeholder")

  feed_dict={
    input_placeholder: input,
    given_data_label_placeholder: given_data_label
  }

  output = inference(input_placeholder)
  loss = loss(output, given_data_label_placeholder)
  training_op = training(loss)

  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    for step in range(1000):
      sess.run(training_op, feed_dict=feed_dict)
      if step % 100 == 0:
        print sess.run(loss, feed_dict=feed_dict)
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

    print 'after learning'
    test_input = numpy.linspace(0.0, 3.0, 31).reshape([31, 1])
    print sess.run(output, feed_dict={input_placeholder:test_input})
