# to report on arxiv paper
import os, argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--s_idx', type=int, help='')
parser.add_argument('--b_idx', type=int, help='')
cfg = parser.parse_args()
sample_idx = cfg.s_idx
b_idx = cfg.b_idx

#plt_style = 'ggplot'
plt_style = 'seaborn-paper'
plt.style.use(plt_style)

sample_idx = 157000
b_idx = 14
#sample_idx = 156000
#b_idx = 11

img_name = 'gp_sample_'+str(sample_idx)+'_'+str(b_idx)+'_'

###############################################################################
# GP option

dirs = [
    '../logs_gp_c/10-17,10:09:58.476452',  # NP
    '../logs_gp_c/10-17,10:10:00.970657',  # ANP
    '../logs_gp_c/10-17,10:09:56.252036',  # SNP
    '../logs_gp_c/10-16,23:58:09.942008',  # ASNP
       ]


###############################################################################

labels = [
          'NP',
          'ANP',
          'SNP',
          'ASNP',
          ]
cm = plt.cm.get_cmap('tab20').colors

# get data
pred_list, std_list, y_list, q_list = [], [], [], []
for direc in dirs:
    with open(os.path.join(direc,'data'+str(sample_idx).zfill(7)+'.pickle'),
              'rb') as f:
        pred_list.append(pkl.load(f))
        std_list.append(pkl.load(f))
        q_list.append(pkl.load(f))
        y_list.append(pkl.load(f))

# plotting
for t in range(50):
    plt.figure(figsize=(7.3*len(dirs), 4.8))
    idx = 1
    for i in range(len(dirs)):
        plt.subplot(1,len(dirs),idx)
        idx +=1

        pred_y = pred_list[i]
        std_y = std_list[i]
        target_y = y_list[i]
        query = q_list[-1]

        (context_x, context_y), target_x = query

        plt.plot(target_x[t][b_idx],
            target_y[t][b_idx], 'k:', linewidth=2,
            label='Targets')

        for _t in range(t):
            alpha = 1.0 - (t-_t) * 0.1
            if alpha < 0.1:
                alpha = 0.1
            if len(context_x[_t]) != 0:
                plt.plot(context_x[_t][b_idx],
                    context_y[_t][b_idx],
                    'bo', markersize=10,
                    alpha=alpha)

        plt.plot(context_x[t][b_idx],
            context_y[t][b_idx],
            'bo', markersize=10,
            label='Old Contexts')

        plt.plot(context_x[t][b_idx],
            context_y[t][b_idx],
            'ko', markersize=10,
            label='Contexts')

        plt.plot(target_x[t][b_idx], pred_y[t][b_idx],
             color=cm[0],
             linestyle='-',
             linewidth=2,
             label='Prediction')

        plt.fill_between(
            target_x[t][b_idx, :, 0],
            pred_y[t][b_idx, :, 0] - std_y[t][b_idx, :, 0],
            pred_y[t][b_idx, :, 0] + std_y[t][b_idx, :, 0],
            alpha=0.2,
            facecolor=cm[1],
            interpolate=True)

        # Make the plot pretty
        plt.yticks([-2, 0, 2], fontsize=13)
        plt.xticks([-4, -2, 0, 2, 4], fontsize=13)
        plt.ylim([-3, 3])
        plt.grid('off')
        axes = plt.gca()
        axes.set_xlim([-4.0,4.0])
        axes.set_ylim([-4.0,4.0])
        if i==0:
            plt.ylabel('y',fontsize=14)
        plt.xlabel('x',fontsize=14)
        plt.title(labels[i],fontsize=15)
        if i==(len(dirs)-1):
            plt.legend(bbox_to_anchor=(0.55, 0.93), loc=2, ncol=1,
                   fontsize=15)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(img_name+str(t).zfill(2)+'.png',bbox_inches='tight')
    plt.close()



