import os
from datetime import datetime
import tensorflow as tf
import numpy as np

from model import *
from gp import *
from plotting import *
from utils import *

# params
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('TRAINING_ITERATIONS', 100000, '@param {type:"number"}')
flags.DEFINE_integer('MAX_CONTEXT_POINTS', 50, '@param {type:"number"}')
flags.DEFINE_integer('PLOT_AFTER', 1000, '@param {type:"number"}')
flags.DEFINE_integer('HIDDEN_SIZE', 128, '@param {type:"number"}')
flags.DEFINE_string('MODEL_TYPE', 'NP', "@param ['NP','ANP']")
flags.DEFINE_string('ATTENTION_TYPE', 'uniform', "@param ['uniform','laplace', 'dot_product', 'multihead']")
flags.DEFINE_boolean('random_kernel_parameters', True, '@param {type:"boolean"}')
flags.DEFINE_string('log_folder', 'logs', '@param {type:"string"}')

# mkdir
log_dir = os.path.join(FLAGS.log_folder, datetime.now().strftime("%m-%d,%H:%M:%S.%f"))
create_directory(log_dir)


summary_ops = []
tf.reset_default_graph()

# tensorboard text
hyperparameters = [tf.convert_to_tensor([key, str(value)]) for key,value in tf.flags.FLAGS.flag_values_dict().items()]
summary_text = tf.summary.text('hyperparameters',tf.stack(hyperparameters))

# tensorboard image
tb_img = tf.placeholder(tf.float32, [None, 480, 640, 3])
summary_img = tf.summary.image('results',tb_img)

# Train dataset
dataset_train = GPCurvesReader(
    batch_size=16, max_num_context=FLAGS.MAX_CONTEXT_POINTS, random_kernel_parameters=FLAGS.random_kernel_parameters)
data_train = dataset_train.generate_curves()

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=1, max_num_context=FLAGS.MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=FLAGS.random_kernel_parameters)
data_test = dataset_test.generate_curves()


# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
latent_encoder_output_sizes = [FLAGS.HIDDEN_SIZE]*4
num_latents = FLAGS.HIDDEN_SIZE
deterministic_encoder_output_sizes= [FLAGS.HIDDEN_SIZE]*4
decoder_output_sizes = [FLAGS.HIDDEN_SIZE]*2 + [2]
use_deterministic_path = True


# ANP with multihead attention
if FLAGS.MODEL_TYPE == 'ANP':
  attention = Attention(rep='mlp', output_sizes=[FLAGS.HIDDEN_SIZE]*2, 
                        att_type=FLAGS.ATTENTION_TYPE)
# NP - equivalent to uniform attention
elif FLAGS.MODEL_TYPE == 'NP':
  attention = Attention(rep='identity', output_sizes=None, att_type=FLAGS.ATTENTION_TYPE)
else:
  raise NameError("MODEL_TYPE not among ['ANP,'NP']")

# Define the model
model = LatentModel(latent_encoder_output_sizes, num_latents,
                    decoder_output_sizes, use_deterministic_path, 
                    deterministic_encoder_output_sizes, attention)

# Define the loss
_, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                 data_train.target_y)

# tensorboard writer
summary_ops.append(tf.summary.scalar('loss', loss))

# Get the predicted mean and variance at the target points for the testing set
mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points)

# Set up the optimizer and train step
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

# Train and plot
with tf.Session() as sess:
  merged_summary = tf.summary.merge(summary_ops)
  writer = tf.summary.FileWriter(log_dir)
  writer.add_summary(sess.run(summary_text))
  sess.run(init)

  for it in range(FLAGS.TRAINING_ITERATIONS):
    sess.run([train_step])

    # Plot the predictions in `PLOT_AFTER` intervals
    if it % FLAGS.PLOT_AFTER == 0:
      summary, loss_value, pred_y, std_y, target_y, whole_query = sess.run(
          [merged_summary, loss, mu, sigma, data_test.target_y, 
           data_test.query])

      writer.add_summary(summary, global_step=it)

      (context_x, context_y), target_x = whole_query
      print('Iteration: {}, loss: {}'.format(it, loss_value))

      # Plot the prediction and the context
      img = plot_functions(log_dir, target_x, target_y, context_x, context_y, pred_y, std_y)
      writer.add_summary(sess.run(summary_img,feed_dict={tb_img:np.reshape(img,[-1,480,640,3])}), global_step=it)
