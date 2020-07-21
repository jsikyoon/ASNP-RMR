import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def plot_functions_1d(len_seq, len_given, len_gen, log_dir, plot_data,
                   hyperparams, h_x_list=None):
  """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
  """
  target_x, target_y, context_x, context_y, pred_y, std = plot_data
  plt.figure(figsize=(6.4, 4.8*(len_seq+len_gen)))
  for t in range(len_seq+len_gen):
      plt.subplot(len_seq+len_gen,1,t+1)
      # Plot everything
      plt.plot(target_x[t][0], target_y[t][0], 'k:', linewidth=2)
      plt.plot(target_x[t][0], pred_y[t][0], 'b', linewidth=2)
      if len(context_x[t]) != 0:
          plt.plot(context_x[t][0], context_y[t][0], 'ko', markersize=10)
      if h_x_list is not None:
          h_y_list = []
          for h_x in h_x_list[t][0]:
            min_val = 10000
            idx = 0
            for i, t_x in enumerate(target_x[t][0]):
                if abs(h_x-t_x) < min_val:
                    min_val = abs(h_x-t_x)
                    idx = i
            h_y_list.append(target_y[t][0][idx])
          plt.plot(h_x_list[t][0],h_y_list, 'ro', markersize=10)
      plt.fill_between(
          target_x[t][0, :, 0],
          pred_y[t][0, :, 0] - std[t][0, :, 0],
          pred_y[t][0, :, 0] + std[t][0, :, 0],
          alpha=0.2,
          facecolor='#65c9f7',
          interpolate=True)

      # Make the plot pretty
      plt.yticks([-4, -2, 0, 2, 4], fontsize=16)
      plt.xticks([-4, -2, 0, 2, 4], fontsize=16)
      plt.grid('off')
      ax = plt.gca()

  plt.savefig(os.path.join(log_dir,'img.png'))
  plt.close()
  image = misc.imread(os.path.join(log_dir,'img.png'),mode='RGB')

  return image

def plot_functions_2d(len_seq, len_given, len_gen, canvas_size, log_dir,
                      plot_data, hyperparams, h_x_list=None):
  """Plots the predicted mean and variance and the context points.
  """
  target_x, target_y, context_x, context_y, pred_y, std = plot_data
  plt.figure(figsize=(4.8*4, 4.8*(len_seq+len_gen)))
  for t in range(len_seq+len_gen):
      tar_canvas = np.zeros((canvas_size,canvas_size,3))
      cont_canvas = np.zeros((canvas_size,canvas_size,3))
      pred_canvas = np.zeros((canvas_size,canvas_size,3))
      std_canvas = np.zeros((canvas_size,canvas_size,3))
      cont_canvas[:,:,2] = 1.0  # default color: blue
      pqset_point = [1.0,0.0,0.0]  # iq color: red

      tar_x = ((target_x[t][0] + 1.0) / 2) * (canvas_size-1) + 0.5
      tar_y = target_y[t][0] + 0.5
      con_x = ((context_x[t][0] + 1.0) / 2) * (canvas_size-1) + 0.5
      con_y = context_y[t][0] + 0.5
      pre_y = pred_y[t][0] + 0.5
      std_y = std[t][0] + 0.5
      if h_x_list is not None:
          h_x = ((h_x_list[t][0] + 1.0) / 2) * (canvas_size-1) + 0.5

      for j in range(len(tar_x)):
          x_loc = int(tar_x[j][0])
          y_loc = int(tar_x[j][1])
          tar_canvas[x_loc][y_loc] = tar_y[j]
          pred_canvas[x_loc][y_loc] = np.clip(pre_y[j],0,1)
          std_canvas[x_loc][y_loc] = np.clip(std_y[j],0,1)
      for j in range(len(con_x)):
          x_loc = int(con_x[j][0])
          y_loc = int(con_x[j][1])
          cont_canvas[x_loc][y_loc] = con_y[j]

      # pqset
      if h_x_list is not None:
        for j in range(len(h_x)):
            x_loc = int(h_x[j][0])
            y_loc = int(h_x[j][1])
            x_loc = np.clip(x_loc, 0, canvas_size-1)
            y_loc = np.clip(y_loc, 0, canvas_size-1)
            cont_canvas[x_loc][y_loc] = pqset_point

      plt.subplot(len_seq+len_gen,4,t*4+1)
      plt.imshow(tar_canvas)
      plt.axis('off')
      plt.subplot(len_seq+len_gen,4,t*4+2)
      plt.imshow(cont_canvas)
      plt.axis('off')
      plt.subplot(len_seq+len_gen,4,t*4+3)
      plt.imshow(pred_canvas)
      plt.axis('off')
      plt.subplot(len_seq+len_gen,4,t*4+4)
      plt.imshow(std_canvas)
      plt.axis('off')

  plt.savefig(os.path.join(log_dir,'img.png'))
  plt.close()
  image = misc.imread(os.path.join(log_dir,'img.png'),mode='RGB')

  return image
