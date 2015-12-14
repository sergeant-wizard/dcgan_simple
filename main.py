import tensorflow as tf
import numpy
import discriminative
import generative

def training(loss):
  with tf.name_scope('d_training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

with tf.Graph().as_default():
  #d_input_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="d_input_placeholder")
  #given_data_label_placeholder = tf.placeholder("float", [None, 2], name="given_placeholder")

  #feed_dict={
  #  d_input_placeholder: discriminative.input,
  #  given_data_label_placeholder: discriminative.given_data_label,
  #}

  #d_output = discriminative.inference(d_input_placeholder)
  #d_loss = discriminative.loss(d_output, given_data_label_placeholder)

  #g_loss = 1 - tf.reduce_sum(tf.nn.softmax(discriminative.inference(g_output)))

  #d_training_op = tf.train.GradientDescentOptimizer(0.01).minimize(
  #  -(d_loss + g_loss),
  #  var_list = discriminative.train_var_list()
  #)

  # g_training_op = tf.train.GradientDescentOptimizer(0.01).minimize(
  #   g_loss,
  #   var_list = ['g_hidden1_weight', 'g_hidden1_bias']
  # )

  aoeu = generative.output
  summary_op = tf.merge_all_summaries()

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    # print generative.feed()
    # print sess.run(aoeu, feed_dict = generative.feed())

    #for step in range(1000):
    #  sess.run(d_training_op, feed_dict=feed_dict)
    #  sess.run(g_training_op, feed_dict=feed_dict)
    #  if step % 100 == 0:
    #    print sess.run(d_loss, feed_dict=feed_dict), sess.run(g_loss, feed_dict=feed_dict)
    #    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    #    summary_writer.add_summary(summary_str, step)

    # print 'after learning'
    # test_input = numpy.linspace(0.0, 3.0, 31).reshape([31, 1])
    # print sess.run(d_output, feed_dict={d_input_placeholder:test_input})
