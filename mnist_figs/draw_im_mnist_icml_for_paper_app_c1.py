import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

matplotlib.rcParams['text.latex.unicode']=True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

img_name = '2d_mnist_app2_'

##############################################
# mnist

sample_idx=900000
b_idx=1
#sample_idx=901000
#b_idx=2
#sample_idx=902000
#b_idx=1

#sample_idx=800000
#b_idx=0

img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'.png'

prefix = '../logs_mnist_a/'
dirs = [
    '07-23,23:53:59.362681', #ANP(h=128)
    '10-22,14:38:15.126605', #SNP(h=128)
    '01-09,11:18:25.847800', #SNPK(h=128, K=25)
    '07-23,23:45:03.191801', #ASNP(h=128, K=25)
       ]
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]

labels = [
          #'NP',
          'ANP',
          'SNP',
          #'SNP-W(K=25)',
          #'SNP-RMR(K=25)',
          #'ASNP-W(K=25)',
          #'ASNP-RMR(K=25)',
          'ASNP-W',
          'ASNP-RMR',
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
        #if idx == len(labels)-1:
        #    h_x_list.append(pkl.load(f))

    if idx == 0:
        canvas_size = int(math.sqrt(len(target[0][0])))

    # [target_x, target_y, context_x, context_y, pred_y, std_y]
    if 'SNP' in labels[idx]:
        data.append(reordering(query, target, pred, std, temporal=True))
    else:
        data.append(reordering(query, target, pred, std, temporal=False))

# plotting
pqset_point = [1.0,0.0,0.0]  # Red
T = np.arange(0,20,2)
plt.figure(figsize=(4.8*(2+len(labels)), 4.8*len(T))) # [context, target(withIm), models] * len(T)
for t_idx, t in enumerate(T):
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        tar_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas[:,:,:] = 1.0  # default color: white
        tar_y = target_y[t][b_idx] + 0.5
        con_x = ((context_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
        con_y = context_y[t][b_idx] + 0.5

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
            #if i == 0:
            tar_canvas[x_loc][y_loc] = 1-tar_y[j][0]
            pred_canvas[x_loc][y_loc] = np.clip(1-pre_y[j][0],0,1)
            std_canvas[x_loc][y_loc] = np.clip(1-std_y[j][0],0,1)

        if i == len(labels)-1:
            for j in range(len(con_x)):
                x_loc = int(con_x[j][0])
                y_loc = int(con_x[j][1])
                if con_y[j][0]!=0:
                    cont_canvas[x_loc][y_loc] = 1-con_y[j][0]
                else:
                    cont_canvas[x_loc][y_loc] = 0
                    cont_canvas[x_loc][y_loc][2] = 1

        # drawing target and context
        if i == len(labels)-1:
            plt.subplot(len(T),2+len(labels),t_idx*(2+len(labels))+1)
            plt.imshow(cont_canvas)
            plt.xticks([])
            plt.yticks([])
            if t_idx == 0:
                plt.title(r'Context',fontsize=40)
            plt.ylabel(r'$'+str(t+1)+'$',fontsize=50)
            plt.subplot(len(T),2+len(labels),t_idx*(2+len(labels))+2)
            plt.imshow(tar_canvas)
            plt.xticks([])
            plt.yticks([])
            if t_idx == 0:
                plt.title(r'Target',fontsize=40)

        plt.subplot(len(T),2+len(labels),t_idx*(2+len(labels))+i+3)
        plt.imshow(pred_canvas)
        plt.xticks([])
        plt.yticks([])
        if t_idx == 0:
            plt.title(r''+labels[i],fontsize=40)

##############################################
# saving
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(img_name, bbox_inches='tight')
plt.close()

