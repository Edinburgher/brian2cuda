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
for profiling_datapoint in ["neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors", "other"]:
  color = colors.pop()
  # legend.append(profiling_datapoint)
  if profiling_datapoint == "other":
    sep_value = openmp_sep_data["last_run_time"] - bottom_openmp_sep_data
    merged_value = openmp_merged_data["last_run_time"] - bottom_openmp_merged_data
  else:
    sep_value = openmp_sep_data[profiling_datapoint]
    merged_value = openmp_merged_data[profiling_datapoint]

  plt.bar("OpenMP seperate", sep_value, color=color, label=profiling_datapoint, bottom=bottom_openmp_sep_data)
  bottom_openmp_sep_data += sep_value
  plt.bar("OpenMP merged", merged_value,color=color, bottom=bottom_openmp_merged_data)
  bottom_openmp_merged_data += merged_value
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
colors.pop()
for profiling_datapoint in ["neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors", "other"]:
  color = colors.pop()
  legend.append(profiling_datapoint)
  if profiling_datapoint == "other":
    sep_value = singlethread_sep_data["last_run_time"] - singlethread_sep_data["neurongroup_stateupdater"] - bottom_singlethread_sep_data
    merged_value = singlethread_merged_data["last_run_time"] - singlethread_merged_data["neurongroup_stateupdater"] - bottom_singlethread_merged_data
  else:
    sep_value = singlethread_sep_data[profiling_datapoint]
    merged_value = singlethread_merged_data[profiling_datapoint]

  plt.bar("Single thread seperate", sep_value,color=color, bottom=bottom_singlethread_sep_data, label=profiling_datapoint)
  bottom_singlethread_sep_data += sep_value
  plt.bar("Single thread merged", merged_value,color=color, bottom=bottom_singlethread_merged_data)
  bottom_singlethread_merged_data += merged_value
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
for profiling_datapoint in ["neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors", "other"]:
  color = colors.pop()
  legend.append(profiling_datapoint)
  if profiling_datapoint == "other":
    sep_value = cuda_sep_data["last_run_time"] - bottom_cuda_sep_data
    merged_value = cuda_merged_data["last_run_time"] - bottom_cuda_merged_data
  else:
    sep_value = cuda_sep_data[profiling_datapoint]
    merged_value = cuda_merged_data[profiling_datapoint]
  plt.bar("CUDA seperate", sep_value,color=color, label=profiling_datapoint, bottom=bottom_cuda_sep_data)
  bottom_cuda_sep_data += sep_value
  plt.bar("CUDA merged", merged_value,color=color, bottom=bottom_cuda_merged_data)
  bottom_cuda_merged_data += merged_value


# plt.yscale('log')
plt.ylabel('seconds')
# plt.xticks(['1','2','4','8','16','32','64','128','256'])
#plt.plot(singlethread_sep_data)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('fig3-cuda.png', bbox_inches='tight')
plt.clf()