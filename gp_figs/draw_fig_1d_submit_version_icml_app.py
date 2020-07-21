##################################################################
# sample drawing
##################################################################

# to report on arxiv paper
import os, argparse
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy import misc
import tensorflow as tf

matplotlib.rcParams['text.latex.unicode']=True
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

sample_idx = 157000
b_idx = 14

img_name = 'gp_app_sample_'+str(sample_idx)+'_'+str(b_idx)+'.png'

###############################################################################
# GP option

dirs = [
    #'../logs_gp_c/10-17,10:09:58.476452',  # NP
    '../logs_gp_c/09-06,18:01:10.611041',  # ANP
    '../logs_gp_c/10-17,10:09:56.252036',  # SNP
    #'../logs_gp_c/11-11,10:29:11.249562',  # ANP
    #'../logs_gp_c/11-11,10:29:15.463796',  # SNP
    #'../logs_gp_c/11-11,18:17:54.618255',  # SNPK(K=inf)
    '../logs_gp_c/01-20,22:30:44.671857',  # SNPK(K=25)
    '../logs_gp_c/09-06,17:59:54.232008',  # ASNP
       ]


###############################################################################

labels = [
          #'NP',
          'ANP',
          'SNP',
          'ASNP-W(K=25)',
          'ASNP-RMR(K=25)',
          ]
cm = plt.cm.get_cmap('tab20').colors
line_color_map = [
                  cm[0],
                  cm[4],
                  cm[6],
                  cm[8],
                  ]
area_color_map = [
                  cm[1],
                  cm[4],
                  cm[6],
                  cm[8],
                ]

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
T = list(range(0,50,5))
plt.figure(figsize=(7.4*4,4.8*len(T)))
idx = 1
for t_idx, t in enumerate(T):
    for i in range(4):
        plt.subplot(len(T),4,idx)
        idx+=1

        pred_y = pred_list[i]
        std_y = std_list[i]
        target_y = y_list[i]
        query = q_list[1]

        (context_x, context_y), target_x = query
        if True:
            for ii in range(len(context_x)):
                for jj in range(len(context_x[ii])):
                    for kk in range(len(context_x[ii][jj])):
                        context_x[ii][jj][kk] = [context_x[ii][jj][kk][0]]
            target_x = np.array(target_x)[:,:,:,:1]

        plt.plot(target_x[t][b_idx],
            target_y[t][b_idx], 'k:', linewidth=2,
            label='Targets')

        for _t in range(t):
            alpha = 1.0 - (t-_t) * 0.1
            if alpha < 0.1:
                alpha = 0.1
            if len(context_x[_t]) != 0:
                #plt.plot(context_x[_t][b_idx],
                plt.plot(context_x[_t][b_idx],
                    context_y[_t][b_idx],
                    'bo', markersize=10,
                    alpha=alpha)

        #plt.plot(context_x[t][b_idx],
        plt.plot(context_x[t][b_idx],
            context_y[t][b_idx],
            'bo', markersize=10,
            label='Old Contexts')

        #plt.plot(context_x[t][b_idx],
        plt.plot(context_x[t][b_idx],
            context_y[t][b_idx],
            'ko', markersize=10,
            label='Contexts')

        #plt.plot(target_x[t][b_idx], pred_y[t][b_idx],
        plt.plot(target_x[t][b_idx], pred_y[t][b_idx],
             color=cm[0],
             linestyle='-',
             linewidth=2,
             label='Prediction')
             #label=labels[i])
        #plt.fill_between(
        plt.fill_between(
            target_x[t][b_idx, :, 0],
            pred_y[t][b_idx, :, 0] - std_y[t][b_idx, :, 0],
            pred_y[t][b_idx, :, 0] + std_y[t][b_idx, :, 0],
            alpha=0.2,
            facecolor=cm[1],
            interpolate=True)

        # Make the plot pretty
        plt.ylim([-3, 3])
        plt.yticks([-2, 0, 2], fontsize=40)
        plt.grid('off')
        axes = plt.gca()
        axes.set_xlim([-4.0,4.0])
        plt.xticks([-4, -2, 0, 2, 4], fontsize=40)
        if i==0:
            plt.ylabel(r'$y(t='+str(t+1)+')$',fontsize=40)
        if t_idx==0:
            plt.title(r'$\mathrm{'+labels[i]+'}$',fontsize=35)
        if t_idx==(len(T)-1):
            plt.xlabel(r'$x$',fontsize=40)
        if i==3 and t_idx==0:
            plt.legend(bbox_to_anchor=(0.45, 0.96), loc=2, ncol=1,
                   fontsize=16)

plt.savefig(img_name,bbox_inches='tight')
plt.close()


