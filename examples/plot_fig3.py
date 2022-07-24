from os import sep
from turtle import color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Truebenchmark.csv")

data = data[(data['has_monitors'] == True)]
data = data.groupby(['device_name','network_count','duration','has_monitors','is_merged','multithreading_type'], as_index=False).mean()
data.sort_values(['network_count'], inplace=True)
sep_data = data[(data['is_merged'] == False)]
len_sep_data=len(sep_data['network_count'])
sep_data = pd.DataFrame(np.repeat(sep_data.values, 9, axis=0), columns=sep_data.columns)
sep_data.sort_values(['multithreading_type'], inplace=True)

sep_data.to_csv('sep_data.csv')
sep_data[
    ['network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
  ] = sep_data[
    ['network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
  ].mul([i for i in 2 ** np.arange(9)]*len_sep_data, axis=0)


data = data[(data['network_count'] == 128)]
sep_data = sep_data[(sep_data['network_count'] == 128)]
sep_data['network_count']=sep_data['network_count'].astype(str)
data['network_count']=data['network_count'].astype(str)

openmp_sep_data = sep_data[(sep_data['multithreading_type'] == 'openmp')]
singlethread_sep_data = sep_data[(sep_data['multithreading_type'] == 'none')]
cuda_sep_data = sep_data[(sep_data['multithreading_type'] == 'GPU')]
singlethread_sep_data.to_csv('singlethread_sep_data.csv')
cuda_sep_data.to_csv('cuda_sep_data.csv')

openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]
legend = []
bottom_openmp_sep_data= 0
bottom_singlethread_sep_data=0
bottom_cuda_sep_data= 0
bottom_openmp_merged_data= 0
bottom_singlethread_merged_data= 0
bottom_cuda_merged_data = 0
colors = ['b','g','r','c','m','y','k','tab:pink','tab:brown']
for profiling_datapoint in ["neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]:
  color = colors.pop()
  # legend.append(profiling_datapoint)
  plt.bar("OpenMP seperate", openmp_sep_data[profiling_datapoint], color=color, label=profiling_datapoint, bottom=bottom_openmp_sep_data)
  bottom_openmp_sep_data += openmp_sep_data[profiling_datapoint]
  plt.bar("OpenMP merged", openmp_merged_data[profiling_datapoint],color=color, bottom=bottom_openmp_merged_data)
  bottom_openmp_merged_data += openmp_merged_data[profiling_datapoint]
# plt.yscale('log')
plt.ylabel('seconds')
# plt.xticks(['1','2','4','8','16','32','64','128','256'])
#plt.plot(singlethread_sep_data)
# Shrink current axis by 20%
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('fig3-openmp.png', bbox_inches='tight')
plt.clf()

legend = []
colors = ['b','g','r','c','m','y','k','tab:pink','tab:brown']
for profiling_datapoint in ["neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]:
  color = colors.pop()
  legend.append(profiling_datapoint)
  plt.bar("Single thread seperate", singlethread_sep_data[profiling_datapoint],color=color, bottom=bottom_singlethread_sep_data, label=profiling_datapoint)
  bottom_singlethread_sep_data += singlethread_sep_data[profiling_datapoint]
  plt.bar("Single thread merged", singlethread_merged_data[profiling_datapoint],color=color, bottom=bottom_singlethread_merged_data)
  bottom_singlethread_merged_data += singlethread_merged_data[profiling_datapoint]
# plt.yscale('log')
plt.ylabel('seconds')
# plt.xticks(['1','2','4','8','16','32','64','128','256'])
#plt.plot(singlethread_sep_data)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('fig3-single-thread.png', bbox_inches='tight')
plt.clf()
legend = []
colors = ['b','g','r','c','m','y','k','tab:pink','tab:brown']
for profiling_datapoint in ["neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]:
  color = colors.pop()
  legend.append(profiling_datapoint)
  plt.bar("CUDA seperate", cuda_sep_data[profiling_datapoint],color=color, label=profiling_datapoint, bottom=bottom_cuda_sep_data)
  bottom_cuda_sep_data += cuda_sep_data[profiling_datapoint]
  plt.bar("CUDA merged", cuda_merged_data[profiling_datapoint],color=color, bottom=bottom_cuda_merged_data)
  bottom_cuda_merged_data += cuda_merged_data[profiling_datapoint]


# plt.yscale('log')
plt.ylabel('seconds')
# plt.xticks(['1','2','4','8','16','32','64','128','256'])
#plt.plot(singlethread_sep_data)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('fig3-cuda.png', bbox_inches='tight')
plt.clf()