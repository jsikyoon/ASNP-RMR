import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, help='')
parser.add_argument('--s_idx', type=int, help='')
parser.add_argument('--b_idx', type=int, help='')
cfg = parser.parse_args()
case = cfg.case
sample_idx = cfg.s_idx
b_idx = cfg.b_idx

# pre-set configuration
fig_folder = 'celeba_figs'

if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

img_name = 'celeba_case'+str(case)+'_'
img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'_2.png'

if case==1:
    # case 1
    dirs = [
    '../logs_celeba_a/10-24,08:06:49.511280',  # NP
    '../logs_celeba_a/11-01,08:35:33.319673',  # ANP
    '../logs_celeba_a/10-24,08:07:18.460748',  # SNP
    '../logs_celeba_a/10-22,14:34:59.347558',  # ASNP
       ]
elif case==2:
    # case 2
    dirs = [
    '../logs_celeba_b/10-24,11:14:09.658519',  # NP
    '../logs_celeba_b/10-29,15:06:54.583922',  # ANP
    '../logs_celeba_b/10-24,11:14:09.516340',  # SNP
    '../logs_celeba_b/10-22,14:34:54.088000',  # ASNP
       ]
elif case==3:
    dirs = [
    '../logs_celeba_c/10-24,13:24:22.169006',  # NP
    '../logs_celeba_c/10-29,08:51:24.805469',  # ANP
    '../logs_celeba_c/10-24,13:24:19.681389',  # SNP
    '../logs_celeba_c/10-22,14:34:48.947854',  # ASNP n_d=25
       ]
else:
    raise NotImplemented




labels = [
          'NP',
          'ANP',
          'SNP',
          'ASNP',
          ]

# get data
data = []
for idx, direc in enumerate(dirs):
    with open(os.path.join(direc,'data'+str(sample_idx).zfill(7)+'.pickle'),
              'rb') as f:
        pred = pkl.load(f)
        std = pkl.load(f)
        query = pkl.load(f)
        target = pkl.load(f)

    if idx == 0:
        canvas_size = int(math.sqrt(len(target[0][0])))

    # [target_x, target_y, context_x, context_y, pred_y, std_y]
    if 'SNP' in labels[idx]:
        data.append(reordering(query, target, pred, std, temporal=True))
    else:
        data.append(reordering(query, target, pred, std, temporal=False))

# plotting
if (case==1) or (case==2):
    length = 20
    ts = list(range(0,18,2))
elif case==3:
    length = 50
    ts = list(range(0,45,5))

plt.figure(figsize=(4.8*(2+len(labels)), 4.8*len(ts)))

for t_idx, t in enumerate(ts):
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        if i == 0:
            tar_canvas = np.ones((canvas_size,canvas_size,3))
            tar_canvas2 = np.zeros((canvas_size,canvas_size,3))
            cont_canvas = np.ones((canvas_size,canvas_size,3))
            #cont_canvas[:,:,2] = 1.0  # default color: blue
            tar_y = target_y[t][b_idx] + 0.5
            con_x = ((context_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
            con_y = context_y[t][b_idx] + 0.5

        pred_canvas = np.ones((canvas_size,canvas_size,3))
        std_canvas = np.ones((canvas_size,canvas_size,3))

        # denormalization
        tar_x = ((target_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
        pre_y = pred_y[t][b_idx] + 0.5
        std_y = std[t][b_idx] + 0.5

        for j in range(len(tar_x)):
            x_loc = int(tar_x[j][0])
            y_loc = int(tar_x[j][1])
            tar_canvas2[x_loc][y_loc] = tar_y[j]
        for j in range(len(tar_x)):
            x_loc = int(tar_x[j][0])
            y_loc = int(tar_x[j][1])
            #if np.sum(tar_y[j])!=0.0:
            if np.sum(tar_canvas2[x_loc,:,:]) != 0 and np.sum(tar_canvas2[:,y_loc,:]) != 0:
                tar_canvas[x_loc][y_loc] = tar_y[j]
                pred_canvas[x_loc][y_loc] = np.clip(pre_y[j],0,1)
                std_canvas[x_loc][y_loc] = np.clip(std_y[j],0,1)

        if i == 0:
            for j in range(len(con_x)):
                x_loc = int(con_x[j][0])
                y_loc = int(con_x[j][1])
                #if np.sum(con_y[j])!=0.0:
                if np.sum(tar_canvas2[x_loc,:,:]) != 0 and np.sum(tar_canvas2[:,y_loc,:]) != 0:
                    cont_canvas[x_loc][y_loc] = con_y[j]
                else:
                    cont_canvas[x_loc][y_loc][0] = 0.0
                    cont_canvas[x_loc][y_loc][1] = 0.0
                    cont_canvas[x_loc][y_loc][2] = 1.0

        # drawing target and context
        if i == 0:
            plt.subplot(len(ts),(2+len(labels)),1+(2+len(labels))*t_idx)
            plt.imshow(cont_canvas)
            #plt.axis('off')
            if t == 0:
                plt.title('Context',fontsize=25)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(len(ts),(2+len(labels)),2+(2+len(labels))*t_idx)
            plt.imshow(tar_canvas)
            #plt.axis('off')
            if t == 0:
                plt.title('Target',fontsize=25)
            plt.xticks([])
            plt.yticks([])

        plt.subplot(len(ts),(2+len(labels)),i+3+(2+len(labels))*t_idx)
        plt.imshow(pred_canvas)
        #plt.axis('off')
        if t == 0:
            plt.title(labels[i],fontsize=25)
        plt.xticks([])
        plt.yticks([])

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(img_name, bbox_inches='tight')
plt.close()

