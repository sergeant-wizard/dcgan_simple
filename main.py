import tensorflow as tf
import numpy
import discriminative
import generative

def training(loss):
  with tf.name_scope('d_training') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

with tf.Graph().as_default():
  d_given_data_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="g_given_data_placeholder")
  g_output_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="g_output_placeholder")

  d_input_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="d_input_placeholder")
  g_input_placeholder = tf.placeholder("float", [None, generative.INPUT_SIZE], name="g_input_placeholder")

  summary_op = tf.merge_all_summaries()

  g_output = generative.inference(g_input_placeholder)
  d_output = discriminative.inference(d_input_placeholder)

  d_output_from_given_data = discriminative.inference(d_given_data_placeholder)
  d_output_from_noise = discriminative.inference(g_output_placeholder)

  d_loss = discriminative.loss(d_output_from_given_data, d_output_from_noise)
  d_train_op = training(d_loss)

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    for step in range(10):
      g_output_eval = sess.run(g_output, feed_dict = {g_input_placeholder: generative.input})
      sess.run(d_train_op, feed_dict = {
        g_output_placeholder: g_output_eval,
        d_given_data_placeholder: discriminative.input
      })

      print sess.run([d_output_from_given_data, d_output_from_noise], feed_dict = {
        g_output_placeholder: g_output_eval,
        d_given_data_placeholder: discriminative.input
      })

    #  sess.run(d_training_op, feed_dict=feed_dict)
    #  sess.run(g_training_op, feed_dict=feed_dict)
    #  if step % 100 == 0:
    #    print sess.run(d_loss, feed_dict=feed_dict), sess.run(g_loss, feed_dict=feed_dict)
    #    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    #    summary_writer.add_summary(summary_str, step)

    # print 'after learning'
    # test_input = numpy.linspace(0.0, 3.0, 31).reshape([31, 1])
    # print sess.run(d_output, feed_dict={d_input_placeholder:test_input})
