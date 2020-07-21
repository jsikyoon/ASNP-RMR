import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

case = 3
sample_idx = 926000
b_idx = 0

img_name = 'celeba1_case'+str(case)+'_'
img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'_2.png'

if case==1:
    # case 1
    dirs = [
       'logs_a/07-23,23:45:01.458419',  # NP
       'logs_a/07-23,23:53:59.362681',  # ANP
       'logs_a/07-23,23:44:55.943355',  # SNP
       'logs_a/07-23,23:45:03.191801',  # ASNP
       ]
elif case==2:
    # case 2
    dirs = [
       'logs_b/07-23,23:45:54.368017',  # NP
       'logs_b/07-23,23:45:59.184011',  # ANP
       'logs_b/07-23,23:46:01.956299',  # SNP
       'logs_b/07-23,23:46:16.857817',  # ASNP
       ]
elif case==3:
    # case 3
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
elif case==3:
    length = 50
#for t in range(length):
T = list(np.arange(0,49,3))

# context, target, NP, ANP, SNP, ASNP
plt.figure(figsize=(4.8*len(T), 4.8*6))
for t_idx, t in enumerate(T):
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        if i == 0:
            tar_canvas = np.ones((canvas_size,canvas_size,3))
            cont_canvas = np.ones((canvas_size,canvas_size,3))
            #cont_canvas[:,:,2] = 1.0  # default color: blue
            cont_canvas[:,:,:] = 1.0  # default color: white
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
            #if i == 0:
            if sum(tar_y[j])!=0.0:
                tar_canvas[x_loc][y_loc] = tar_y[j]
                pred_canvas[x_loc][y_loc] = np.clip(pre_y[j],0,1)
                std_canvas[x_loc][y_loc] = np.clip(std_y[j],0,1)

        if i == 0:
            for j in range(len(con_x)):
                x_loc = int(con_x[j][0])
                y_loc = int(con_x[j][1])
                if sum(con_y[j])!=0.0:
                    cont_canvas[x_loc][y_loc] = con_y[j]
                else:
                    cont_canvas[x_loc][y_loc][0] = 0.0
                    cont_canvas[x_loc][y_loc][1] = 0.0
                    cont_canvas[x_loc][y_loc][2] = 1.0

        # drawing target and context
        if i == 0:
            plt.subplot(6,len(T),1+t_idx)
            plt.imshow(cont_canvas)
            plt.xticks([])
            plt.yticks([])
            #plt.axis('off')
            plt.title(str(t),fontsize=50)
            if t_idx==0:
                plt.ylabel('Context', fontsize=50)
            plt.subplot(6,len(T),1+t_idx+len(T))
            plt.imshow(tar_canvas)
            plt.xticks([])
            plt.yticks([])
            #plt.axis('off')
            if t_idx==0:
                plt.ylabel('Target',fontsize=50)
                #plt.title('Target',fontsize=25)

        plt.subplot(6,len(T),1+t_idx+len(T)*(i+2))
        plt.imshow(pred_canvas)
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
        if t_idx==0:
            plt.ylabel(labels[i],fontsize=50)
            #plt.title(labels[i],fontsize=25)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(img_name, bbox_inches='tight')
plt.close()

