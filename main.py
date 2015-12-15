import tensorflow as tf
import numpy
import discriminative
import generative

def training(loss, learning_rate):
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  return train_step

with tf.Graph().as_default():
  # placeholders
  d_given_data_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="g_given_data_placeholder")
  g_output_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="g_output_placeholder")

  d_input_placeholder = tf.placeholder("float", [None, discriminative.INPUT_SIZE], name="d_input_placeholder")
  g_input_placeholder = tf.placeholder("float", [None, generative.INPUT_SIZE], name="g_input_placeholder")

  d_hidden1_weight_placeholder = tf.placeholder("float", [discriminative.INPUT_SIZE, discriminative.HIDDEN_UNIT_SIZE])
  d_hidden1_bias_placeholder = tf.placeholder("float", [discriminative.HIDDEN_UNIT_SIZE])
  d_output_weight_placeholder = tf.placeholder("float", [discriminative.HIDDEN_UNIT_SIZE, 1])
  d_output_bias_placeholder = tf.placeholder("float", [1])

  # inference output
  g_output = generative.inference(g_input_placeholder)

  with tf.variable_scope('d_params') as scope:
    d_output_from_given_data = discriminative.trainable_inference(d_given_data_placeholder)
    scope.reuse_variables()
    d_output_from_noise_for_dtrain = discriminative.trainable_inference(g_output_placeholder)

  d_output_from_noise_for_gtrain = discriminative.inference(
    g_output,
    d_hidden1_weight_placeholder,
    d_hidden1_bias_placeholder,
    d_output_weight_placeholder,
    d_output_bias_placeholder
  )

  # loss
  d_loss_1, d_loss_2 = discriminative.loss(d_output_from_given_data, d_output_from_noise_for_dtrain)
  g_loss = tf.reduce_sum(tf.log(1 - d_output_from_noise_for_gtrain))

  # training
  d_train_op = training(-(d_loss_1 + d_loss_2), 0.01)
  g_train_op = training(g_loss, 0.001)

  # misc
  with tf.variable_scope('d_params', reuse = True) as scope:
    d_params = discriminative.get_train_params()
  summary_op = tf.merge_all_summaries()
  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('data', graph_def=sess.graph_def)
    sess.run(init)

    num_steps = 256
    for step in range(num_steps):
      g_output_eval = sess.run(g_output, feed_dict = {g_input_placeholder: generative.input})
      sess.run(d_train_op, feed_dict = {
        g_output_placeholder: g_output_eval,
        d_given_data_placeholder: discriminative.input
      })

      d_train_params_eval = sess.run(d_params)
      sess.run(g_train_op, feed_dict = {
        g_input_placeholder: generative.input,
        d_hidden1_weight_placeholder: d_train_params_eval[0],
        d_hidden1_bias_placeholder:   d_train_params_eval[1],
        d_output_weight_placeholder:  d_train_params_eval[2],
        d_output_bias_placeholder:    d_train_params_eval[3]
      })

      feed_dict = {
        g_output_placeholder: g_output_eval,
        d_given_data_placeholder: discriminative.input,
        g_input_placeholder: generative.input,
        d_hidden1_weight_placeholder: d_train_params_eval[0],
        d_hidden1_bias_placeholder:   d_train_params_eval[1],
        d_output_weight_placeholder:  d_train_params_eval[2],
        d_output_bias_placeholder:    d_train_params_eval[3]
      }

      if step % (num_steps / 8) == 0:
        print 'loss:'
        print sess.run([d_loss_1, d_loss_2, g_loss], feed_dict = feed_dict)

    test_input = numpy.linspace(-3.0, 3.0, 61).reshape([61, 1])
    print 'discriminator distribution:'
    print sess.run(d_output_from_given_data, feed_dict={d_given_data_placeholder: test_input})
    print 'generated:'
    print sess.run(g_output, feed_dict={g_input_placeholder: generative.input})
