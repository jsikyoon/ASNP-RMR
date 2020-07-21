##################################################################
# graph drawing
##################################################################
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#plt_style = 'ggplot'
plt_style = 'seaborn-paper'
plt.style.use(plt_style)

img_name = 'gp_graph.png'

###############################################################################
# task a option

dirs_a = [
    '../logs_gp_a/10-17,15:46:54.040597',  # NP
    '../logs_gp_a/10-17,15:46:53.837892',  # ANP
    '../logs_gp_a/10-17,15:46:57.296676',  # SNP
    '../logs_gp_a/10-17,15:46:53.145704',  # ASNP
    '../logs_gp_a/11-07,12:05:24.432223',  # ASNPwoImg(K=1)
    ]

###############################################################################
# task b option

dirs_b = [
    '../logs_gp_b/10-17,15:46:57.525708',  # NP
    '../logs_gp_b/10-17,15:47:01.631708',  # ANP
    '../logs_gp_b/10-17,15:47:08.129047',  # SNP
    '../logs_gp_b/10-17,15:46:55.106220',  # ASNP
    '../logs_gp_b/11-06,17:19:09.386418',  # ASNPwoImg(K=1)
     ]

###############################################################################
# task c option

dirs_c = [
    '../logs_gp_c/10-17,10:09:58.476452',  # NP
    '../logs_gp_c/10-17,10:10:00.970657',  # ANP
    '../logs_gp_c/10-17,10:09:56.252036',  # SNP
    '../logs_gp_c/10-16,23:58:09.942008',  # ASNP
    '../logs_gp_c/11-06,17:19:10.271166',  # ASNPwoImg(K=1)
     ]

###############################################################################

smoothing = 100

# get data from Tensorboard logs
vals_a, iters_a = [], []
min_iter_a = 10000000000
for dir_name in dirs_a:
    print(dir_name)
    summ = [EventAccumulator(dir_name).Reload()]

    tags = summ[0].Tags()['scalars']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summ]):
            out[tag].append([e.value for e in events])
    steps = [e.step for e in summ[0].Scalars(tag)]

    vals_a.append(out)
    iters_a.append(steps)

    if steps[-1] < min_iter_a:
        min_iter_a = steps[-1]

vals_b, iters_b = [], []
min_iter_b = 10000000000
for dir_name in dirs_b:
    print(dir_name)
    summ = [EventAccumulator(dir_name).Reload()]

    tags = summ[0].Tags()['scalars']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summ]):
            out[tag].append([e.value for e in events])
    steps = [e.step for e in summ[0].Scalars(tag)]

    vals_b.append(out)
    iters_b.append(steps)

    if steps[-1] < min_iter_b:
        min_iter_b = steps[-1]

vals_c, iters_c = [], []
min_iter_c = 10000000000
for dir_name in dirs_c:
    print(dir_name)
    summ = [EventAccumulator(dir_name).Reload()]

    tags = summ[0].Tags()['scalars']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summ]):
            out[tag].append([e.value for e in events])
    steps = [e.step for e in summ[0].Scalars(tag)]

    vals_c.append(out)
    iters_c.append(steps)

    if steps[-1] < min_iter_c:
        min_iter_c = steps[-1]

# smoothing
tag_list_ab = []
for i in range(20):
    tag_list_ab.append('TargetNLL/nll_'+str(i))
tag_list_c = []
for i in range(50):
    tag_list_c.append('TargetNLL/nll_'+str(i))

data_a = []
for val, iteration in zip(vals_a, iters_a):
    _data = []
    for tag in tag_list_ab:
        iter_idx = iteration.index(min_iter_a)
        #_data.append(np.mean(val[tag][(iter_idx-smoothing):iter_idx]))
        _data.append(np.mean(val[tag][-smoothing:]))
    data_a.append(_data)

data_b = []
for val, iteration in zip(vals_b, iters_b):
    _data = []
    for tag in tag_list_ab:
        iter_idx = iteration.index(min_iter_b)
        #_data.append(np.mean(val[tag][(iter_idx-smoothing):iter_idx]))
        _data.append(np.mean(val[tag][-smoothing:]))
    data_b.append(_data)

data_c = []
for val, iteration in zip(vals_c, iters_c):
    _data = []
    for tag in tag_list_c:
        iter_idx = iteration.index(min_iter_c)
        #_data.append(np.mean(val[tag][(iter_idx-smoothing):iter_idx]))
        _data.append(np.mean(val[tag][-smoothing:]))
    data_c.append(_data)

# labels
labels = [
          'NP',
          'ANP',
          'SNP',
          'ASNP',
          'ASNPwoImg(K=1)',
          'ASNPwoImg(K=3)',
          'ASNPwoImg(K=inf)',
          ]

cm = plt.cm.get_cmap('tab20').colors
line_color_map = [
                  cm[0],
                  cm[4],
                  cm[6],
                  cm[8],
                  cm[10],
                 ]

linestyle = ['-','-','-','-','-']

# plotting
top = 0.09
bottom = 0.14
height = 1 - top - bottom
width = 0.278
wspace = 0.048
left1 = 0.066
left2 = left1 + width + wspace
wspace = 0.048
left3 = left2 + width + wspace

rec1 = [left1, bottom, width, height]
rec2 = [left2, bottom, width, height]
rec3 = [left3, bottom, width, height]


plt.figure(figsize=(6.4*3, 4.8))
ax1 = plt.axes(rec1)
#plt.subplot(1,4,1)
for i in range(len(dirs_a)):
    #plt.plot(list(range(1,len(tag_list_ab)+1)),
    ax1.plot(list(range(1,len(tag_list_ab)+1)),
             data_a[i],
             color=line_color_map[i],
             label=labels[i],
             linewidth=2,
             linestyle=linestyle[i])
    print("task a")
    print(labels[i])
    d_str = ""
    for j in range(1,len(tag_list_ab)+1):
        d_str+= "("
        d_str+=str(j)+","
        d_str+= str(data_a[i][j-1])
        d_str+= ")"
    print(d_str)

# make plot pretty
#plt.grid(True)
ax1.grid(True)
axes = plt.gca()
#axes = ax1.gca()
#plt.yticks(list(range(-0.2,1.4,0.2)),fontsize=12)
plt.yticks(fontsize=18)
#ax1.set_yticklabels(list(range(-0.2,1.4,0.2)),fontsize=13)
#axes.set_ylim([-0.5, 1.3])
plt.xticks(list(range(1,26,3)),fontsize=18)
#ax1.set_xticklabels(list(range(1,26,2)),fontsize=13)
#axes.set_xlim([1.0, 20.0])
ax1.set_xlim([1.0, 20.0])
#plt.xlabel('Time',fontsize=14)
ax1.set_xlabel('Time',fontsize=19)
#plt.ylabel('Target NLL',fontsize=14)
ax1.set_ylabel('Target NLL',fontsize=19)
#legend = plt.legend(bbox_to_anchor=(0.08, 0.99), title="Model (setting)", loc=2, ncol=1, fontsize=14)
#legend = ax1.legend(bbox_to_anchor=(0.08, 1.02), title="Model", loc=2, ncol=1, fontsize=19)
#plt.setp(legend.get_title(),fontsize=19)
#ax1.setp(legend.get_title(),fontsize=14)
ax1.set_title('Scenario (a)', fontsize=20)

ax2 = plt.axes(rec2)
for i in range(len(dirs_b)):
    #plt.plot(list(range(1,len(tag_list_ab)+1)),
    ax2.plot(list(range(1,len(tag_list_ab)+1)),
             data_b[i],
             color=line_color_map[i],
             label=labels[i],
             linewidth=2,
             linestyle=linestyle[i])
    print("task b")
    print(labels[i])
    d_str = ""
    for j in range(1,len(tag_list_ab)+1):
        d_str+= "("
        d_str+=str(j)+","
        d_str+= str(data_b[i][j-1])
        d_str+= ")"
    print(d_str)

# make plot pretty
#plt.grid(True)
ax2.grid(True)
axes = plt.gca()
#axes = ax2.gca()
#plt.yticks(list(range(-0.2,1.4,0.2)),fontsize=12)
plt.yticks(fontsize=18)
#ax2.yticks(fontsize=13)
#axes.set_ylim([-0.5, 1.3])
#plt.xticks(list(range(1,50,5)),fontsize=18)
plt.xticks(list(range(1,26,3)),fontsize=18)
#ax2.set_xticklabels(list(range(1,50,3)),fontsize=13)
axes.set_xlim([1.0, 20.0])
#plt.xlabel('Time',fontsize=14)
ax2.set_xlabel('Time',fontsize=19)
#plt.ylabel('Target NLL',fontsize=14)
#legend = plt.legend(bbox_to_anchor=(0.08, 0.99), title="Model (setting)", loc=2, ncol=1, fontsize=14)
#legend = ax2.legend(bbox_to_anchor=(0.08, 1.02), title="Model", loc=2, ncol=1, fontsize=19)
#plt.setp(legend.get_title(),fontsize=19)
#ax2.setp(legend.get_title(),fontsize=14)
ax2.set_title('Scenario (b)', fontsize=20)


# plotting
#plt.subplot(1,4,2)
ax3 = plt.axes(rec3)
for i in range(len(dirs_c)):
    #plt.plot(list(range(1,len(tag_list_c)+1)),
    ax3.plot(list(range(1,len(tag_list_c)+1)),
             data_c[i],
             color=line_color_map[i],
             label=labels[i],
             linewidth=2,
             linestyle=linestyle[i])
    print("task c")
    print(labels[i])
    d_str = ""
    for j in range(1,len(tag_list_c)+1,3):
        d_str+= "("
        d_str+=str(j)+","
        d_str+= str(data_c[i][j-1])
        d_str+= ")"
    print(d_str)

# make plot pretty
#plt.grid(True)
ax3.grid(True)
axes = plt.gca()
#axes = ax2.gca()
#plt.yticks(list(range(-0.2,1.4,0.2)),fontsize=12)
plt.yticks(fontsize=18)
#ax2.yticks(fontsize=13)
#axes.set_ylim([-0.5, 1.3])
plt.xticks(list(range(1,50,5)),fontsize=18)
#ax2.set_xticklabels(list(range(1,50,3)),fontsize=13)
axes.set_xlim([1.0, 50.0])
#plt.xlabel('Time',fontsize=14)
ax3.set_xlabel('Time',fontsize=19)
#plt.ylabel('Target NLL',fontsize=14)
#legend = plt.legend(bbox_to_anchor=(0.08, 0.99), title="Model (setting)", loc=2, ncol=1, fontsize=14)
legend = ax3.legend(bbox_to_anchor=(0.48, 1.02), title="Model", loc=2, ncol=1, fontsize=19)
plt.setp(legend.get_title(),fontsize=19)
#ax2.setp(legend.get_title(),fontsize=14)
ax3.set_title('Scenario (c)', fontsize=20)

plt.savefig(img_name)
plt.close()

