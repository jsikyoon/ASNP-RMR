import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

img_name = '2d_introfig_'

plt.figure(figsize=(4.8*6, 4.8*2))

##############################################
# mnist
sample_idx=900000
b_idx=1
t = 40
img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'_'+str(t).zfill(2)+'_'

prefix = 'logs_mnist_c/'
dirs = [
    #'10-24,13:24:22.169006', # NP(h=128)
    '07-25,10:24:15.632419', # ANP(h=128)
    '10-22,16:12:26.898386', # SNP(h=128)
    '11-24,19:49:00.297868', # SNP-Att(h=128,K=inf)
    '07-25,10:24:09.806928', # SNP-TMRA(h=128,K=25)
       ]
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]

labels = [
          'ANP',
          'SNP',
          'SNP-Att',
          'SNP-TMRA',
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
for i in range(len(labels)):

    target_x, target_y, context_x, context_y, pred_y, std = data[i]

    if i == 0:
        tar_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas = np.ones((canvas_size,canvas_size,3))
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
        if i == 0:
            tar_canvas[x_loc][y_loc] = 1-tar_y[j][0]
        pred_canvas[x_loc][y_loc] = np.clip(1-pre_y[j][0],0,1)
        std_canvas[x_loc][y_loc] = np.clip(1-std_y[j][0],0,1)

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
        plt.subplot(2,6,1)
        plt.imshow(cont_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title('Context',fontsize=25)
        plt.subplot(2,6,7)
        plt.imshow(tar_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title('Target',fontsize=25)

    if i < 2:
        plt.subplot(2,6,i+2)
    else:
        plt.subplot(2,6,i+6)
    plt.imshow(pred_canvas)
    plt.xticks([])
    plt.yticks([])
    plt.title(labels[i],fontsize=25)


##############################################
# celeba
sample_idx=910000
b_idx=0
t = 40
img_name += str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'_'+str(t).zfill(2)+'.png'

prefix = 'logs_celeba_c/'
dirs = [
    #'10-24,13:24:22.169006', # NP(h=128)
    '10-29,08:51:24.805469', # ANP(h=128)
    '10-24,13:24:19.681389', # SNP(h=128)
    '12-28,08:10:36.113834', # SNP-Att(h=128,K=inf)
    '10-22,14:34:48.947854', # SNP-TMRA(h=128,K=25)
       ]
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]

labels = [
          'ANP',
          'SNP',
          'SNP-Att',
          'SNP-TMRA',
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
for i in range(len(labels)):

    target_x, target_y, context_x, context_y, pred_y, std = data[i]

    if i == 0:
        tar_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas[:,:,2] = 1.0  # default color: blue
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
        plt.subplot(2,6,4)
        plt.imshow(cont_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title('Context',fontsize=25)
        plt.subplot(2,6,10)
        plt.imshow(tar_canvas)
        plt.xticks([])
        plt.yticks([])
        plt.title('Target',fontsize=25)

    if i < 2:
        plt.subplot(2,6,i+5)
    else:
        plt.subplot(2,6,i+9)
    plt.imshow(pred_canvas)
    plt.xticks([])
    plt.yticks([])
    plt.title(labels[i],fontsize=25)

##############################################
# saving
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(img_name, bbox_inches='tight')
plt.close()

