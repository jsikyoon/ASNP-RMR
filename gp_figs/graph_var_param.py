# to report on arxiv paper
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

img_name = 'gp_var_param_graph.png'
length=20
dirs = [
    '../logs_gp_a_re/10-17,15:46:54.040597',  # NP(n=128)
    '../logs_gp_a_re/11-06,17:25:05.183065',  # NP(n=512)
    '../logs_gp_a_re/10-17,15:46:53.837892',  # ANP(n=128)
    '../logs_gp_a_re/11-06,17:25:09.696126',  # ANP(n=512)
    '../logs_gp_a_re/10-17,15:46:57.296676',  # SNP(n=128)
    '../logs_gp_a_re/11-06,17:25:03.213940',  # SNP(n=512)
    '../logs_gp_a_re/10-17,15:46:53.145704',  # ASNP
     ]

smoothing = 100

# get data from Tensorboard logs
vals, iters = [], []
min_iter = 10000000000
for dir_name in dirs:
    print(dir_name)
    summ = [EventAccumulator(dir_name).Reload()]

    tags = summ[0].Tags()['scalars']

    out = defaultdict(list)
    steps = []

    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summ]):
            out[tag].append([e.value for e in events])
    steps = [e.step for e in summ[0].Scalars(tag)]

    vals.append(out)
    iters.append(steps)

    if steps[-1] < min_iter:
        min_iter = steps[-1]

# smoothing
tag_list = []
for i in range(length):
    tag_list+=['TargetNLL/nll_'+str(i)]

data = []
for val, iteration in zip(vals, iters):
    _data = []
    for tag in tag_list:
        iter_idx = iteration.index(min_iter)
        #_data.append(np.mean(val[tag][(iter_idx-smoothing):iter_idx]))
        _data.append(np.mean(val[tag][-smoothing:]))
    data.append(_data)

# labels
labels = [
          'NP(n=128)',
          'NP(n=512)',
          'ANP(n=128)',
          'ANP(n=512)',
          'SNP(n=128)',
          'SNP(n=512)',
          'ASNP(n=128)',
          ]

cm = plt.cm.get_cmap('tab20').colors
line_color_map = [cm[0], cm[1],
                  cm[2], cm[3],
                  cm[4], cm[5],
                  cm[6]
                  ]

# plotting
plt.figure(figsize=(6.4, 4.8))
for i in range(len(dirs)):
    plt.plot(list(range(1,len(tag_list)+1)),
             data[i],
             color=line_color_map[i],
             label=labels[i],
             linewidth=2)

    print(labels[i])
    d_str = ""
    for j in range(1,len(tag_list)+1):
        d_str+="("
        d_str+=str(j)+","
        d_str+=str(data[i][j-1])
        d_str+=")"
    print(d_str)


# make plot pretty
plt.grid(True)
axes = plt.gca()
plt.yticks(fontsize=13)
plt.xticks(list(range(1,length,2)),fontsize=13)
plt.xlabel('Time',fontsize=14)
plt.ylabel('Target NLL',fontsize=14)
legend = plt.legend(bbox_to_anchor=(0.08, 0.99), title="Model", loc=2, ncol=1, fontsize=15)
plt.setp(legend.get_title(),fontsize=15)
plt.savefig(img_name, bbox_inches='tight')
plt.close()


