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
#parser = argparse.ArgumentParser()
#parser.add_argument('--s_idx', type=int, help='')
#parser.add_argument('--b_idx', type=int, help='')
#cfg = parser.parse_args()
#sample_idx = cfg.s_idx
#b_idx = cfg.b_idx

sample_idx = 156000
b_idx = 11

#plt_style = 'ggplot'
#plt_style = 'seaborn-paper'
#plt.style.use(plt_style)

#sample_idx = 320000
#b_idx = 5
#img_name = 'gp_sample1_'+str(sample_idx)+'_'+str(b_idx)+'.png'
img_name = 'gp_sample_'+str(sample_idx)+'_'+str(b_idx)+'.png'

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
plt.figure(figsize=(7.4*2,4.8*2))
#for t in range(50):
for t in range(32,33):
    #plt.figure(figsize=(5.5*2, 4.8))
    idx = 1
    for i in range(4):
        plt.subplot(2,2,i+1)

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
        #plt.yticks([-2, 0, 2], fontsize=18)
        #plt.xticks([-4, -2, 0, 2, 4], fontsize=18)
        plt.ylim([-3, 3])
        plt.grid('off')
        axes = plt.gca()
        #ax.set_yticklabels([-2, 0, 2], fontsize=13)
        #ax.set_xticklabels([-4, -2, 0, 2, 4], fontsize=13)
        #ax.set_ylim([-3, 3])
        #ax.grid('off')
        #axes = ax.gca()
        axes.set_xlim([-4.0,4.0])
        #axes.set_ylim([-4.0,4.0])
        #ax.set_xlim([-4.0,4.0])
        #ax.set_ylim([-4.0,4.0])
        #plt.ylabel('y',fontsize=14)
        #plt.xlabel('x',fontsize=14)
        if i > 1:
            #plt.xlabel('x',fontsize=19)
            plt.xlabel(r'$x$',fontsize=30)
            plt.xticks([-4, -2, 0, 2, 4], fontsize=30)
        else:
            plt.xticks([-4, -2, 0, 2, 4], fontsize=30)
            #plt.xticks([], fontsize=18)
        #plt.title(labels[i],fontsize=20)
        plt.title(r'$\mathrm{'+labels[i]+'}$',fontsize=25)
        if i == 0 or i == 2:
            #plt.ylabel('y',fontsize=19)
            plt.ylabel(r'$y$',fontsize=30)
            plt.yticks([-2, 0, 2], fontsize=30)
        else:
            plt.yticks([-2, 0, 2], fontsize=30)
            #plt.yticks([], fontsize=18)
        #if idx ==3:
        #plt.legend(bbox_to_anchor=(0.47, 0.38), loc=2, ncol=1,
        #    ax.legend(bbox_to_anchor=(0.47, 0.45), loc=2, ncol=1,
        #           fontsize=19)

    #plt.subplots_adjust(wspace=0.05, hspace=0.1)
    #plt.subplots_adjust(hspace=0.1)
    #plt.subplot_tool()
    #plt.savefig(img_name+str(t).zfill(2)+'.png',dpi=500)
    #plt.savefig(img_name,bbox_inches='tight',dpi=500)
    plt.subplots_adjust(wspace=0.15, hspace=0.27)
    plt.savefig(img_name,bbox_inches='tight')
    plt.close()

    #if i == 0:
    #    plt.title('NP', fontsize=12)
    #if i == 1:
    #    plt.title('Laplace ANP', fontsize=12)
    #if i == 2:
    #    plt.title('Multihead ANP', fontsize=12)
    #if i == 3:
    #    plt.title('SNP', fontsize=12)

    """
    if i == 1 or i == 3:
        axes.set_yticklabels([])
    if i == 0 or i == 1:
        axes.set_xticklabels([])
    if 'gp' and 'case1' in img_name:
        axes.set_xlim([-4.0,4.0])
        #axes.set_ylim([-5.0,5.0])
        axes.set_ylim([-3.0,3.0])
    if i == 0 or i == 2:
        plt.ylabel('y',fontsize=12)
    #if i == 1 or i == 3:
    #    plt.gca().axes.get_yaxis().set_visible(False)
    if i == 2 or i == 3:
        plt.xlabel('x',fontsize=12)
    if i==1:
        plt.legend(bbox_to_anchor=(0.55, 0.93), loc=2, ncol=1,
                   fontsize=12)
    """

