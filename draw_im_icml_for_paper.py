import os, argparse, math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf
from utils import reordering

img_name = '2d_long_'
T = np.arange(0,50,6)
labels = [
          'NP',
          'ANP',
          'SNP',
          'SNPK',
          'ASNP',
          ]
plt.figure(figsize=(4.8*len(T)*2, 4.8*(2+len(labels)))) # [context, target(withIm), models] * len(T)

##############################################
# mnist
sample_idx=929000
b_idx=0
img_name += 'mnist_'+str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)

prefix = 'logs_mnist_c/'
dirs = [
    '07-25,10:24:11.220550', # NP(h=128)
    '07-25,10:24:15.632419', # ANP(h=128)
    '10-22,16:12:26.898386', # SNP(h=128)
    '12-28,08:07:05.227516', # SNP-Att(K=inf)
    '07-25,10:24:09.806928', # SNP-RMRA(h=128,K=25)
       ]
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]


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
for t_idx, t in enumerate(T):
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        #if i == 0:
        tar_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas[:,:,:] = 1.0  # default color: white
        tar_y = target_y[t][b_idx] + 0.5
        con_x = ((context_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
        con_y = context_y[t][b_idx] + 0.5
        #if i == len(labels)-1:
        #    h_x = ((h_x_list[0][t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5

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

        # pqset
        #if i == len(labels)-1:
        #    for j in range(len(h_x)):
        #        x_loc = int(h_x[j][0])
        #        y_loc = int(h_x[j][1])
        #        x_loc = np.clip(x_loc, 0, canvas_size-1)
        #        y_loc = np.clip(y_loc, 0, canvas_size-1)
        #        tar_canvas[x_loc][y_loc] = pqset_point # drawing on target image

        if i == len(labels)-1:
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
        if i == len(labels)-1:
            plt.subplot(2+len(labels),len(T)*2,t_idx+1)
            plt.imshow(cont_canvas)
            plt.xticks([])
            plt.yticks([])
            if t_idx == 0:
                plt.ylabel('Context',fontsize=40)
            plt.title(str(t),fontsize=50)
            plt.subplot(2+len(labels),len(T)*2,len(T)*2+t_idx+1)
            plt.imshow(tar_canvas)
            plt.xticks([])
            plt.yticks([])
            if t_idx == 0:
                plt.ylabel('Target/Imgs',fontsize=40)

        #plt.subplot(1,4,3)
        #plt.imshow(im_canvas)
        #plt.xticks([])
        #plt.yticks([])
        #plt.title('ImgPoints',fontsize=25)

        plt.subplot(2+len(labels),len(T)*2,len(T)*2*2+len(T)*2*i+t_idx+1)
        plt.imshow(pred_canvas)
        plt.xticks([])
        plt.yticks([])
        if t_idx == 0:
            plt.ylabel(labels[i],fontsize=40)

##############################################
# celeba
draw_im = False
sample_idx=926000
b_idx=0
dirs = [
    '10-24,13:24:22.169006', # NP(h=128)
    '10-29,08:51:24.805469', # ANP(h=128)
    '10-24,13:24:19.681389', # SNP(h=128)
    '12-28,08:10:36.113834', # SNP-Att(K=inf)
    '10-22,14:34:48.947854', # SNP-RMRA(h=128,K=25)
       ]
prefix = 'logs_celeba_c/'
for i in range(len(dirs)):
    dirs[i] = prefix+dirs[i]
img_name += '_celeba_'+str(sample_idx).zfill(7)+'_'+str(b_idx).zfill(2)+'.png'

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
        if idx == len(labels)-1:
            h_x_list.append(pkl.load(f))

    if idx == 0:
        canvas_size = int(math.sqrt(len(target[0][0])))

    # [target_x, target_y, context_x, context_y, pred_y, std_y]
    if 'SNP' in labels[idx]:
        data.append(reordering(query, target, pred, std, temporal=True))
    else:
        data.append(reordering(query, target, pred, std, temporal=False))

# plotting
pqset_point = [1.0,0.0,0.0]  # Red
for t_idx, t in enumerate(T):
    for i in range(len(labels)):

        target_x, target_y, context_x, context_y, pred_y, std = data[i]

        #if i == 0:
        tar_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas = np.ones((canvas_size,canvas_size,3))
        cont_canvas[:,:,:] = 1.0  # default color: white
        tar_y = target_y[t][b_idx] + 0.5
        con_x = ((context_x[t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5
        con_y = context_y[t][b_idx] + 0.5
        if i == len(labels)-1:
            h_x = ((h_x_list[0][t][b_idx] + 1.0) / 2) * (canvas_size-1) + 0.5

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
            if sum(tar_y[j])!=0.0:
                tar_canvas[x_loc][y_loc] = tar_y[j]
                pred_canvas[x_loc][y_loc] = np.clip(pre_y[j],0,1)
                std_canvas[x_loc][y_loc] = np.clip(std_y[j],0,1)

        # pqset
        if draw_im:
            if i == len(labels)-1:
                for j in range(len(h_x)):
                    x_loc = int(h_x[j][0])
                    y_loc = int(h_x[j][1])
                    x_loc = np.clip(x_loc, 0, canvas_size-1)
                    y_loc = np.clip(y_loc, 0, canvas_size-1)
                    tar_canvas[x_loc][y_loc] = pqset_point # drawing on target image
                    # to make denser
                    x_loc_p = np.clip(x_loc-1, 0, canvas_size-1)
                    y_loc_p = np.clip(y_loc, 0, canvas_size-1)
                    tar_canvas[x_loc_p][y_loc_p] = pqset_point # drawing on target image
                    x_loc_p = np.clip(x_loc, 0, canvas_size-1)
                    y_loc_p = np.clip(y_loc-1, 0, canvas_size-1)
                    tar_canvas[x_loc_p][y_loc_p] = pqset_point # drawing on target image
                    x_loc_p = np.clip(x_loc+1, 0, canvas_size-1)
                    y_loc_p = np.clip(y_loc, 0, canvas_size-1)
                    tar_canvas[x_loc_p][y_loc_p] = pqset_point # drawing on target image
                    x_loc_p = np.clip(x_loc, 0, canvas_size-1)
                    y_loc_p = np.clip(y_loc+1, 0, canvas_size-1)
                    tar_canvas[x_loc_p][y_loc_p] = pqset_point # drawing on target image

        if i == len(labels)-1:
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
        if i == len(labels)-1:
            plt.subplot(2+len(labels),len(T)*2,len(T)+t_idx+1)
            plt.imshow(cont_canvas)
            plt.xticks([])
            plt.yticks([])
            #if t_idx == 0:
            #    plt.ylabel('Context',fontsize=40)
            plt.title(str(t),fontsize=50)
            plt.subplot(2+len(labels),len(T)*2,len(T)*2+len(T)+t_idx+1)
            plt.imshow(tar_canvas)
            plt.xticks([])
            plt.yticks([])
            #if t_idx == 0:
            #    plt.ylabel('Target/Imgs',fontsize=40)

        #plt.subplot(1,4,3)
        #plt.imshow(im_canvas)
        #plt.xticks([])
        #plt.yticks([])
        #plt.title('ImgPoints',fontsize=25)

        plt.subplot(2+len(labels),len(T)*2,len(T)*2*2+len(T)*2*i+len(T)+t_idx+1)
        plt.imshow(pred_canvas)
        plt.xticks([])
        plt.yticks([])
        #if t_idx == 0:
        #    plt.ylabel(labels[i],fontsize=40)

##############################################
# saving
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig(img_name, bbox_inches='tight')
plt.close()
#
