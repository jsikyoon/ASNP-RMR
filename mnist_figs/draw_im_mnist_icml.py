import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

img_name = 'im_mnist_figs/2d_im_mnist_'

##############################################
# mnist
sample_idx=900000
b_idx=0
img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'_'

prefix = '../logs_bck/logs_mnist_c/'
dirs = [
    '10-19,08:54:15.865026', # SNP-TMRA(h=128,K=25)
       ]
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]

labels = [
          'SNP-TMRA',
          ]

# get data
data = []
h_x_list = []
for idx, direc in enumerate(dirs):
    with open(os.path.join(direc,'data'+str(sample_idx).zfill(7)+'.pickle'),
              'rb') as f:
        pred = pkl.load(f)
        std = pkl.load(f)
        query = pkl.load(f)
        target = pkl.load(f)
        hyperparam = pkl.load(f)
        h_x_list.append(pkl.load(f))

    if idx == 0:
        canvas_size = int(math.sqrt(len(target[0][0])))

    # [target_x, target_y, context_x, context_y, pred_y, std_y]
    if 'SNP' in labels[idx]:
        data.append(reordering(query, target, pred, std, temporal=True))
    else:
        data.append(reordering(query, target, pred, std, temporal=False))

# plotting
pqset_point = [1.0,0.0,0.0]
for t in range(50):
    plt.figure(figsize=(4.8*4, 4.8*1))
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        if i == 0:
            tar_canvas = np.ones((canvas_size,canvas_size,3))
            cont_canvas = np.ones((canvas_size,canvas_size,3))
            cont_canvas[:,:,:] = 1.0  # default color: white
            tar_y = target_y[t][b_idx] + 0.5
            con_x = ((context_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
            con_y = context_y[t][b_idx] + 0.5
            h_x = ((h_x_list[i][t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5

        pred_canvas = np.ones((canvas_size,canvas_size,3))
        std_canvas = np.ones((canvas_size,canvas_size,3))
        im_canvas = np.ones((canvas_size,canvas_size,3))

        # denormalization
        tar_x = ((target_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
        pre_y = pred_y[t][b_idx] + 0.5
        std_y = std[t][b_idx] + 0.5

        for j in range(len(tar_x)):
            x_loc = int(tar_x[j][0])
            y_loc = int(tar_x[j][1])
            if i == 0:
                tar_canvas[x_loc][y_loc] = 1-tar_y[j][0]
            pred_canvas[x_loc][y_loc] = np.clip(1-pre_y[j][0],0,1)
            std_canvas[x_loc][y_loc] = np.clip(1-std_y[j][0],0,1)

        # pqset
        for j in range(len(h_x)):
            x_loc = int(h_x[j][0])
            y_loc = int(h_x[j][1])
            x_loc = np.clip(x_loc, 0, canvas_size-1)
            y_loc = np.clip(y_loc, 0, canvas_size-1)
            im_canvas[x_loc][y_loc] = pqset_point

        if i == 0:
            for j in range(len(con_x)):
                x_loc = int(con_x[j][0])
                y_loc = int(con_x[j][1])
                #cont_canvas[x_loc][y_loc][:2] = con_y[j][0]
                if con_y[j][0]!=0:
                    cont_canvas[x_loc][y_loc] = 1-con_y[j][0]
                else:
                    cont_canvas[x_loc][y_loc] = 0
                    cont_canvas[x_loc][y_loc][2] = 1

        # drawing target and context
        if i == 0:
            plt.subplot(1,4,1)
            plt.imshow(cont_canvas)
            plt.xticks([])
            plt.yticks([])
            plt.title('Context',fontsize=25)
            plt.subplot(1,4,2)
            plt.imshow(tar_canvas)
            plt.xticks([])
            plt.yticks([])
            plt.title('Target',fontsize=25)

        plt.subplot(1,4,3)
        plt.imshow(im_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title('ImgPoints',fontsize=25)

        plt.subplot(1,4,4)
        plt.imshow(pred_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title(labels[i],fontsize=25)

    ##############################################
    # saving
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(img_name+str(t).zfill(2)+'.png', bbox_inches='tight')
    plt.close()

